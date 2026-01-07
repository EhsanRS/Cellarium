"""
DataManager - Core data handling for Cellarium.

This module provides the DataManager singleton class that handles:
- Loading AnnData from .h5ad files
- Precomputing embeddings for fast access
- Managing cell subsets for pages
- Caching gene expression data

Design Principles:
1. Load AnnData once at startup, keep in memory
2. Precompute and cache embedding coordinates as numpy arrays
3. Use views (not copies) for subsets when possible
4. LRU cache for frequently accessed gene expression data
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd

from cellarium.config import Config
from cellarium.db.schema import update_root_page_count


@dataclass
class EmbeddingData:
    """Precomputed embedding coordinates for fast access."""

    name: str  # e.g., "X_umap", "X_pca"
    coordinates: np.ndarray  # Shape: (n_cells, n_dims)
    dimensions: List[str]  # e.g., ["UMAP1", "UMAP2"]

    @property
    def n_dims(self) -> int:
        """Number of dimensions in this embedding."""
        return self.coordinates.shape[1]


@dataclass
class SubsetInfo:
    """Information about a cell subset."""

    page_id: str
    cell_indices: np.ndarray  # Integer indices into parent
    parent_page_id: Optional[str]  # None for root
    n_cells: int


class LRUCache:
    """Simple LRU cache for gene expression data."""

    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get item from cache, moving to end (most recently used)."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: np.ndarray) -> None:
        """Add item to cache, evicting oldest if necessary."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class DataManager:
    """
    Singleton-like data manager for AnnData operations.

    This class manages the loaded AnnData object and provides efficient
    access to embeddings, gene expression, and metadata. It uses caching
    and precomputation to ensure fast response times for visualization.

    Usage:
        dm = DataManager(config)
        coords = dm.get_embedding_coords("X_umap", page_id="root")
        expr = dm.get_gene_expression("CD3D", page_id="root")
    """

    _instance: Optional["DataManager"] = None

    def __new__(cls, config: Optional[Config] = None):
        """Singleton pattern - return existing instance if available."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Config] = None):
        """Initialize the DataManager with configuration."""
        if self._initialized:
            return

        if config is None:
            raise ValueError("Config required for first initialization")

        self.config = config
        self._adata: Optional[ad.AnnData] = None
        self._embeddings: Dict[str, EmbeddingData] = {}
        self._subsets: Dict[str, SubsetInfo] = {}
        self._expression_cache: LRUCache = LRUCache(config.expression_cache_size)

        self._load_data()
        self._precompute_embeddings()
        self._initialized = True

    def _load_data(self) -> None:
        """Load AnnData from h5ad file."""
        print(f"Loading AnnData from {self.config.data_path}...")

        # Load fully into memory for medium-sized datasets (50K-500K cells)
        # backed='r' mode would be slower for repeated access patterns
        self._adata = ad.read_h5ad(self.config.data_path)

        # Create root subset representing all cells
        self._subsets["root"] = SubsetInfo(
            page_id="root",
            cell_indices=np.arange(self._adata.n_obs),
            parent_page_id=None,
            n_cells=self._adata.n_obs,
        )

        # Update root page count in database
        update_root_page_count(self.config.db_path, self._adata.n_obs)

        print(f"Loaded {self._adata.n_obs:,} cells × {self._adata.n_vars:,} genes")

    def _precompute_embeddings(self) -> None:
        """
        Precompute embedding coordinates as contiguous numpy arrays.

        This is critical for performance - accessing obsm repeatedly
        through AnnData is slower than direct numpy array access.
        """
        for key in self._adata.obsm.keys():
            coords = np.ascontiguousarray(self._adata.obsm[key])
            n_dims = coords.shape[1]

            # Generate dimension names
            base_name = key.replace("X_", "").upper()
            dim_names = [f"{base_name}{i+1}" for i in range(n_dims)]

            self._embeddings[key] = EmbeddingData(
                name=key,
                coordinates=coords,
                dimensions=dim_names,
            )

        print(f"Precomputed {len(self._embeddings)} embeddings: {list(self._embeddings.keys())}")

    # ==================== Properties ====================

    @property
    def adata(self) -> ad.AnnData:
        """Direct access to AnnData (use sparingly)."""
        return self._adata

    @property
    def n_cells(self) -> int:
        """Total number of cells in dataset."""
        return self._adata.n_obs

    @property
    def n_genes(self) -> int:
        """Total number of genes in dataset."""
        return self._adata.n_vars

    @property
    def cell_ids(self) -> pd.Index:
        """Cell identifiers (obs_names)."""
        return self._adata.obs_names

    @property
    def gene_names(self) -> List[str]:
        """Gene names (var_names) as list."""
        return list(self._adata.var_names)

    @property
    def available_embeddings(self) -> List[str]:
        """List of available embedding keys."""
        return list(self._embeddings.keys())

    @property
    def obs_columns(self) -> List[str]:
        """Available cell metadata columns."""
        return list(self._adata.obs.columns)

    @property
    def layers(self) -> List[str]:
        """Available data layers."""
        return list(self._adata.layers.keys()) if self._adata.layers else []

    @property
    def var_columns(self) -> List[str]:
        """Available gene metadata columns (for gene symbols)."""
        return list(self._adata.var.columns)

    @property
    def has_raw(self) -> bool:
        """Whether raw data is available."""
        return self._adata.raw is not None

    @property
    def raw_n_genes(self) -> int:
        """Number of genes in raw data (if available)."""
        return self._adata.raw.n_vars if self._adata.raw is not None else 0

    def get_gene_symbol_column_options(self) -> List[str]:
        """Get var columns that could contain gene symbols (string type)."""
        options = []
        for col in self._adata.var.columns:
            if self._adata.var[col].dtype == object or str(self._adata.var[col].dtype).startswith('str'):
                options.append(col)
        return options

    def _looks_like_ensembl_id(self, value: str) -> bool:
        """Check if a value looks like an Ensembl ID."""
        if not isinstance(value, str):
            return False
        # Ensembl IDs: ENSG (human), ENSMUSG (mouse), ENSDARG (zebrafish), etc.
        return value.startswith(('ENS', 'ERCC', 'LINC')) or (len(value) > 10 and value[:4].isalpha() and value[4:].replace('.', '').isdigit())

    def _looks_like_gene_symbol(self, value: str) -> bool:
        """Check if a value looks like a gene symbol."""
        if not isinstance(value, str):
            return False
        # Gene symbols are typically 1-15 chars, alphanumeric with possible hyphens
        # They don't start with ENS and aren't very long numeric strings
        if self._looks_like_ensembl_id(value):
            return False
        if len(value) > 20:
            return False
        # Most gene symbols are uppercase letters with optional numbers/hyphens
        return bool(value) and not value.isdigit()

    def detect_var_names_type(self) -> str:
        """
        Detect whether var_names are gene symbols or identifiers (like Ensembl IDs).

        Returns:
            'symbols' if var_names appear to be gene symbols
            'ensembl' if var_names appear to be Ensembl IDs
            'unknown' if can't determine
        """
        # Sample some var_names to check
        sample_size = min(100, len(self._adata.var_names))
        sample_names = self._adata.var_names[:sample_size]

        ensembl_count = sum(1 for name in sample_names if self._looks_like_ensembl_id(name))

        if ensembl_count > sample_size * 0.8:  # >80% look like Ensembl
            return 'ensembl'
        elif ensembl_count < sample_size * 0.2:  # <20% look like Ensembl
            return 'symbols'
        return 'unknown'

    def detect_gene_symbols_column(self) -> Optional[str]:
        """
        Auto-detect the best column containing gene symbols.

        Returns:
            Column name if found, None otherwise
        """
        # Only need to detect if var_names aren't already symbols
        if self.detect_var_names_type() == 'symbols':
            return None

        candidate_cols = self.get_gene_symbol_column_options()
        if not candidate_cols:
            return None

        # Common column names for gene symbols
        common_names = ['gene_symbol', 'gene_symbols', 'symbol', 'symbols',
                        'gene_name', 'gene_names', 'name', 'gene', 'external_gene_name',
                        'gene_short_name', 'Symbol', 'GeneName', 'SYMBOL', 'NAME']

        # First, try exact matches with common names
        for name in common_names:
            if name in candidate_cols:
                return name

        # Try case-insensitive match
        for col in candidate_cols:
            if col.lower() in [n.lower() for n in common_names]:
                return col

        # Score each column by how much it looks like gene symbols
        best_col = None
        best_score = 0

        for col in candidate_cols:
            values = self._adata.var[col].dropna().astype(str)
            if len(values) == 0:
                continue

            sample_size = min(100, len(values))
            sample_values = values.iloc[:sample_size]

            # Count how many look like gene symbols
            symbol_count = sum(1 for v in sample_values if self._looks_like_gene_symbol(v))
            score = symbol_count / sample_size

            if score > best_score and score > 0.5:  # At least 50% should look like symbols
                best_score = score
                best_col = col

        return best_col

    def get_gene_display_names(self, use_raw: bool = True) -> List[str]:
        """
        Get gene names for display in dropdowns, preferring symbols over IDs.

        Args:
            use_raw: If True and raw is available, use raw.var_names

        Returns:
            List of gene names (symbols if available, otherwise var_names)
        """
        # Determine data source
        if use_raw and self._adata.raw is not None:
            adata_source = self._adata.raw
        else:
            adata_source = self._adata

        # Check if var_names are already symbols
        var_names_type = self.detect_var_names_type()

        if var_names_type == 'symbols':
            return list(adata_source.var_names)

        # Try to find a symbols column
        symbols_col = self.detect_gene_symbols_column()

        if symbols_col and symbols_col in adata_source.var.columns:
            # Return symbols, filtering out NaN/empty
            symbols = adata_source.var[symbols_col].fillna('')
            # Return non-empty symbols, fall back to var_name if empty
            result = []
            for var_name, symbol in zip(adata_source.var_names, symbols):
                if symbol and str(symbol).strip():
                    result.append(str(symbol))
                else:
                    result.append(var_name)
            return result

        # Fall back to var_names
        return list(adata_source.var_names)

    def get_gene_name_mapping(self, use_raw: bool = True) -> Dict[str, str]:
        """
        Get mapping from display name (symbol) to var_name.

        Returns:
            Dict mapping display names to actual var_names for indexing
        """
        if use_raw and self._adata.raw is not None:
            adata_source = self._adata.raw
        else:
            adata_source = self._adata

        var_names_type = self.detect_var_names_type()

        if var_names_type == 'symbols':
            # var_names are already symbols, map to themselves
            return {name: name for name in adata_source.var_names}

        symbols_col = self.detect_gene_symbols_column()

        if symbols_col and symbols_col in adata_source.var.columns:
            symbols = adata_source.var[symbols_col].fillna('')
            mapping = {}
            for var_name, symbol in zip(adata_source.var_names, symbols):
                if symbol and str(symbol).strip():
                    mapping[str(symbol)] = var_name
                else:
                    mapping[var_name] = var_name
            return mapping

        return {name: name for name in adata_source.var_names}

    def resolve_gene_name(self, gene: str, gene_symbols_column: Optional[str] = None, use_raw: bool = False) -> str:
        """
        Resolve a gene display name to the actual var_name.

        Args:
            gene: Gene name (could be symbol or var_name)
            gene_symbols_column: Column in var containing gene symbols
            use_raw: Whether to use raw data

        Returns:
            The actual var_name to use for indexing
        """
        adata = self._adata.raw if use_raw and self._adata.raw else self._adata

        # First check if it's directly in var_names
        if gene in adata.var_names:
            return gene

        # If gene_symbols_column specified, try to map
        if gene_symbols_column and gene_symbols_column in adata.var.columns:
            mask = adata.var[gene_symbols_column] == gene
            if mask.any():
                return adata.var_names[mask][0]

        # Not found
        raise ValueError(f"Gene not found: {gene}")

    # ==================== Embedding Access ====================

    def get_embedding_coords(
        self,
        embedding_key: str,
        page_id: str = "root",
        dims: Tuple[int, int] = (0, 1),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embedding coordinates for a page/subset.

        Args:
            embedding_key: Key in obsm (e.g., 'X_umap', 'X_pca')
            page_id: Page/subset ID ('root' for all cells)
            dims: Tuple of dimension indices to return

        Returns:
            Tuple of (x, y) coordinate arrays ready for Plotly
        """
        if embedding_key not in self._embeddings:
            raise ValueError(f"Unknown embedding: {embedding_key}")

        coords = self._embeddings[embedding_key].coordinates

        # Apply subset mask if not root
        if page_id != "root":
            subset = self._subsets.get(page_id)
            if subset is None:
                raise ValueError(f"Unknown page: {page_id}")
            coords = coords[subset.cell_indices]

        return coords[:, dims[0]], coords[:, dims[1]]

    def get_embedding_dims(self, embedding_key: str) -> List[str]:
        """Get dimension names for an embedding."""
        if embedding_key not in self._embeddings:
            raise ValueError(f"Unknown embedding: {embedding_key}")
        return self._embeddings[embedding_key].dimensions

    def get_embedding_dataframe(
        self,
        embedding_key: str,
        page_id: str = "root",
        include_metadata: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get embedding coordinates with optional metadata for plotting.

        Optimized for Plotly: returns DataFrame with x, y columns
        plus any requested metadata columns for coloring.

        Args:
            embedding_key: Key in obsm
            page_id: Page/subset ID
            include_metadata: List of obs columns to include

        Returns:
            DataFrame with 'x', 'y', 'cell_index' and optional metadata
        """
        embedding = self._embeddings[embedding_key]
        coords = embedding.coordinates
        indices = self._get_subset_indices(page_id)

        df = pd.DataFrame({
            "x": coords[indices, 0],
            "y": coords[indices, 1],
            "cell_index": indices,  # Original index for selection tracking
        })

        if include_metadata:
            for col in include_metadata:
                if col in self._adata.obs.columns:
                    df[col] = self._adata.obs[col].values[indices]

        return df

    # ==================== Expression Data Access ====================

    def get_gene_expression(
        self,
        gene: str,
        page_id: str = "root",
        layer: Optional[str] = None,
        use_raw: bool = False,
        gene_symbols_column: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get expression values for a gene, with caching.

        Performance strategy:
        - Cache entire gene vectors (not subsets) to maximize reuse
        - LRU eviction when cache exceeds limit

        Args:
            gene: Gene name (or symbol if gene_symbols_column specified)
            page_id: Page/subset ID
            layer: Data layer to use (None for X)
            use_raw: Whether to use adata.raw (contains all genes, not just HVGs)
            gene_symbols_column: Column in var containing gene symbols (for mapping)

        Returns:
            1D numpy array of expression values
        """
        # Resolve gene name (handles symbol mapping)
        resolved_gene = self.resolve_gene_name(gene, gene_symbols_column, use_raw)

        cache_key = f"{resolved_gene}:{layer or 'X'}:{'raw' if use_raw else 'X'}"

        # Check cache first
        cached = self._expression_cache.get(cache_key)
        if cached is None:
            # Select data source
            if use_raw and self._adata.raw is not None:
                adata_source = self._adata.raw
                # raw doesn't have layers, always use X
                data = adata_source.X
            else:
                adata_source = self._adata
                if layer and layer in self._adata.layers:
                    data = self._adata.layers[layer]
                else:
                    data = self._adata.X

            # Get gene index
            try:
                gene_idx = adata_source.var_names.get_loc(resolved_gene)
            except KeyError:
                raise ValueError(f"Gene not found: {resolved_gene}")

            # Extract and cache (handle sparse matrices)
            expr = data[:, gene_idx]
            if hasattr(expr, "toarray"):
                expr = expr.toarray().flatten()
            else:
                expr = np.asarray(expr).flatten()

            self._expression_cache.set(cache_key, expr)
            cached = expr

        # Apply subset
        indices = self._get_subset_indices(page_id)
        return cached[indices]

    def get_genes_expression(
        self,
        genes: List[str],
        page_id: str = "root",
        layer: Optional[str] = None,
        use_raw: bool = False,
        gene_symbols_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get expression values for multiple genes.

        Args:
            genes: List of gene names
            page_id: Page/subset ID
            layer: Data layer to use
            use_raw: Whether to use adata.raw
            gene_symbols_column: Column in var containing gene symbols

        Returns:
            DataFrame with genes as columns
        """
        data = {}
        for gene in genes:
            try:
                data[gene] = self.get_gene_expression(gene, page_id, layer, use_raw, gene_symbols_column)
            except ValueError:
                continue  # Skip genes not found

        return pd.DataFrame(data)

    # ==================== Metadata Access ====================

    def get_obs_data(
        self,
        columns: List[str],
        page_id: str = "root",
    ) -> pd.DataFrame:
        """Get cell metadata for specified columns."""
        indices = self._get_subset_indices(page_id)
        return self._adata.obs.iloc[indices][columns].copy()

    def get_obs_column(
        self,
        column: str,
        page_id: str = "root",
    ) -> np.ndarray:
        """Get values for a single obs column."""
        indices = self._get_subset_indices(page_id)
        return self._adata.obs[column].values[indices]

    def get_obs_unique_values(self, column: str) -> np.ndarray:
        """Get unique values for an obs column."""
        if column not in self._adata.obs.columns:
            raise ValueError(f"Column not found: {column}")
        return self._adata.obs[column].unique()

    def is_categorical(self, column: str) -> bool:
        """Check if an obs column is categorical."""
        if column not in self._adata.obs.columns:
            return False
        dtype = self._adata.obs[column].dtype
        return dtype.name == "category" or dtype == object

    # ==================== Subset Management ====================

    def create_subset(
        self,
        page_id: str,
        cell_indices: np.ndarray,
        parent_page_id: str = "root",
    ) -> SubsetInfo:
        """
        Create a new subset from selected cell indices.

        IMPORTANT: cell_indices should be indices into the PARENT subset,
        not absolute indices into the full dataset.

        Args:
            page_id: Unique ID for the new subset
            cell_indices: Indices of cells to include (relative to parent)
            parent_page_id: ID of the parent page

        Returns:
            SubsetInfo for the new subset
        """
        # Convert to absolute indices if parent is not root
        if parent_page_id != "root":
            parent_subset = self._subsets.get(parent_page_id)
            if parent_subset is None:
                raise ValueError(f"Parent page not found: {parent_page_id}")
            absolute_indices = parent_subset.cell_indices[cell_indices]
        else:
            absolute_indices = cell_indices

        subset = SubsetInfo(
            page_id=page_id,
            cell_indices=np.asarray(absolute_indices, dtype=np.int64),
            parent_page_id=parent_page_id,
            n_cells=len(absolute_indices),
        )

        self._subsets[page_id] = subset
        return subset

    def get_subset_adata(self, page_id: str) -> ad.AnnData:
        """
        Get AnnData view for a subset.

        Note: Returns a VIEW, not a copy. Do not modify!
        For exports, use .copy() on the result.
        """
        indices = self._get_subset_indices(page_id)
        return self._adata[indices]

    def delete_subset(self, page_id: str) -> bool:
        """Remove a subset from memory."""
        if page_id in self._subsets and page_id != "root":
            del self._subsets[page_id]
            return True
        return False

    def get_subset_info(self, page_id: str) -> Optional[SubsetInfo]:
        """Get info about a subset."""
        return self._subsets.get(page_id)

    def _get_subset_indices(self, page_id_or_indices) -> np.ndarray:
        """Get cell indices for a page/subset or from direct indices.

        Args:
            page_id_or_indices: Either a page_id string OR a list/array of cell indices.
                               If None or "root", returns all cells.
        """
        # Handle None or "root" -> all cells
        if page_id_or_indices is None or page_id_or_indices == "root":
            return np.arange(self._adata.n_obs)

        # Handle direct cell indices (list or array)
        if isinstance(page_id_or_indices, (list, np.ndarray)):
            return np.asarray(page_id_or_indices)

        # Handle page_id string
        subset = self._subsets.get(page_id_or_indices)
        if subset is None:
            raise ValueError(f"Unknown page: {page_id_or_indices}")
        return subset.cell_indices

    # ==================== Statistics ====================

    def get_subset_stats(self, page_id: str) -> Dict[str, Any]:
        """
        Get statistics for a subset.

        Returns dict with:
        - n_cells: Number of cells
        - percent_of_total: Percentage of total cells
        - categorical_counts: Dict of value counts for categorical columns
        """
        indices = self._get_subset_indices(page_id)
        n_cells = len(indices)

        stats = {
            "n_cells": n_cells,
            "percent_of_total": (n_cells / self.n_cells) * 100,
            "categorical_counts": {},
        }

        # Count categorical distributions (limit to avoid performance issues)
        for col in self._adata.obs.columns[:10]:  # First 10 columns
            if self.is_categorical(col):
                counts = self._adata.obs[col].iloc[indices].value_counts()
                stats["categorical_counts"][col] = counts.to_dict()

        return stats

    # ==================== Gene Expression Analysis ====================

    def compute_top_expressed_genes(
        self,
        cell_indices: List[int],
        top_n: int = 100,
        use_raw: bool = True,
        gene_symbols_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute average expression across selected cells and return top expressed genes.

        This method efficiently computes mean expression for ALL genes in the selected
        cells, then returns the top N genes by average expression.

        Args:
            cell_indices: List of cell indices to analyze
            top_n: Number of top genes to return (default 100)
            use_raw: Whether to use raw counts (more complete gene set)
            gene_symbols_column: Column in var containing gene symbols

        Returns:
            DataFrame with columns: ['gene_symbol', 'gene_id', 'mean_expression', 'pct_expressing']
            Sorted by mean_expression descending
        """
        if not cell_indices:
            return pd.DataFrame(columns=['gene_symbol', 'gene_id', 'mean_expression', 'pct_expressing'])

        cell_indices = np.asarray(cell_indices)

        # Select data source
        if use_raw and self._adata.raw is not None:
            data = self._adata.raw.X
            var_names = self._adata.raw.var_names
            var_df = self._adata.raw.var
        else:
            data = self._adata.X
            var_names = self._adata.var_names
            var_df = self._adata.var

        # Subset to selected cells
        subset_data = data[cell_indices, :]

        # Handle sparse matrices efficiently
        if hasattr(subset_data, "toarray"):
            # For sparse: compute mean and pct_expressing using sparse operations
            # Convert to dense only for the final subset
            subset_dense = subset_data.toarray()
        else:
            subset_dense = np.asarray(subset_data)

        # Compute statistics
        n_cells = len(cell_indices)
        mean_expression = np.mean(subset_dense, axis=0)
        pct_expressing = (np.sum(subset_dense > 0, axis=0) / n_cells) * 100

        # Build results DataFrame
        results = pd.DataFrame({
            'gene_id': var_names,
            'mean_expression': mean_expression,
            'pct_expressing': pct_expressing,
        })

        # Add gene symbols if available
        if gene_symbols_column and gene_symbols_column in var_df.columns:
            results['gene_symbol'] = var_df[gene_symbols_column].values
        elif self.detect_var_names_type() == 'symbols':
            results['gene_symbol'] = var_names
        else:
            # Try auto-detected column
            detected_col = self.detect_gene_symbols_column()
            if detected_col and detected_col in var_df.columns:
                results['gene_symbol'] = var_df[detected_col].values
            else:
                results['gene_symbol'] = var_names

        # Reorder columns
        results = results[['gene_symbol', 'gene_id', 'mean_expression', 'pct_expressing']]

        # Sort by mean expression and take top N
        results = results.sort_values('mean_expression', ascending=False).head(top_n)

        # Round for display
        results['mean_expression'] = results['mean_expression'].round(4)
        results['pct_expressing'] = results['pct_expressing'].round(1)

        return results.reset_index(drop=True)

    # ==================== Crosstab / Dendrogram Analysis ====================

    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns in obs."""
        categorical_cols = []
        for col in self._adata.obs.columns:
            dtype = self._adata.obs[col].dtype
            if dtype.name == "category" or dtype == object:
                # Check if it has a reasonable number of categories (not continuous-like)
                n_unique = self._adata.obs[col].nunique()
                if n_unique <= 200:  # Reasonable limit for categorical
                    categorical_cols.append(col)
        return categorical_cols

    def get_available_representations(self) -> List[str]:
        """Get available representations in obsm for dendrogram computation."""
        return list(self._adata.obsm.keys())

    def get_default_representation(self) -> Optional[str]:
        """Get default representation for dendrogram (prefer X_scVI if available)."""
        obsm_keys = self.get_available_representations()
        # Prefer X_scVI, then X_pca
        for preferred in ["X_scVI", "X_scvi", "X_pca", "X_PCA"]:
            if preferred in obsm_keys:
                return preferred
        return obsm_keys[0] if obsm_keys else None

    def get_or_compute_dendrogram(
        self,
        groupby: str,
        use_rep: Optional[str] = None,
        force_recompute: bool = False,
    ) -> Dict[str, Any]:
        """
        Get existing dendrogram or compute a new one for the groupby column.

        Args:
            groupby: Column to group by (e.g., 'leiden_0.5')
            use_rep: Representation to use (e.g., 'X_scVI', 'X_pca'). Auto-detected if None.
            force_recompute: Force recomputation even if exists

        Returns:
            Dict with dendrogram info including:
            - 'linkage': scipy linkage matrix
            - 'categories_ordered': ordered category names
            - 'dendrogram_info': full scanpy dendrogram output
        """
        import scanpy as sc
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist

        dendrogram_key = f"dendrogram_{groupby}"

        # Check if dendrogram exists and matches
        if not force_recompute and dendrogram_key in self._adata.uns:
            dendro_info = self._adata.uns[dendrogram_key]
            # Verify it has what we need
            if "linkage" in dendro_info and "categories_ordered" in dendro_info:
                return {
                    "linkage": dendro_info["linkage"],
                    "categories_ordered": dendro_info["categories_ordered"],
                    "dendrogram_info": dendro_info,
                    "computed": False,
                }

        # Need to compute dendrogram
        if use_rep is None:
            use_rep = self.get_default_representation()

        # Compute dendrogram using scanpy
        try:
            sc.tl.dendrogram(
                self._adata,
                groupby=groupby,
                use_rep=use_rep,
                key_added=dendrogram_key,
            )
        except Exception as e:
            # If scanpy fails, compute manually
            return self._compute_dendrogram_manual(groupby, use_rep)

        dendro_info = self._adata.uns[dendrogram_key]
        return {
            "linkage": dendro_info["linkage"],
            "categories_ordered": dendro_info["categories_ordered"],
            "dendrogram_info": dendro_info,
            "computed": True,
            "use_rep": use_rep,
        }

    def _compute_dendrogram_manual(
        self,
        groupby: str,
        use_rep: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Manually compute dendrogram when scanpy fails.

        Uses mean representation per group and hierarchical clustering.
        """
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist

        # Get representation data
        if use_rep and use_rep in self._adata.obsm:
            rep_data = self._adata.obsm[use_rep]
        else:
            # Fall back to X (may be large)
            rep_data = self._adata.X
            if hasattr(rep_data, "toarray"):
                rep_data = rep_data.toarray()

        # Get group assignments
        groups = self._adata.obs[groupby]
        categories = groups.cat.categories if hasattr(groups, "cat") else groups.unique()

        # Compute mean representation per group
        group_means = []
        valid_categories = []
        for cat in categories:
            mask = groups == cat
            if mask.sum() > 0:
                mean_rep = np.mean(rep_data[mask], axis=0)
                group_means.append(mean_rep)
                valid_categories.append(cat)

        group_means = np.array(group_means)

        # Compute linkage
        if len(valid_categories) > 1:
            distances = pdist(group_means)
            Z = linkage(distances, method="average")
            # Get ordered categories
            order = leaves_list(Z)
            categories_ordered = [valid_categories[i] for i in order]
        else:
            Z = np.array([])
            categories_ordered = list(valid_categories)

        return {
            "linkage": Z,
            "categories_ordered": categories_ordered,
            "dendrogram_info": {"linkage": Z, "categories_ordered": categories_ordered},
            "computed": True,
            "use_rep": use_rep,
        }

    def compute_crosstab_data(
        self,
        groupby: str,
        compare_col: str,
        cell_indices: Optional[List[int]] = None,
        use_rep: Optional[str] = None,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute crosstab between two categorical columns with dendrogram ordering.

        Args:
            groupby: Primary grouping column (rows, will have dendrogram)
            compare_col: Comparison column (columns)
            cell_indices: Optional subset of cells to use
            use_rep: Representation for dendrogram computation
            normalize: Whether to normalize rows to sum to 1

        Returns:
            Dict with:
            - 'crosstab': DataFrame with crosstab values
            - 'crosstab_normalized': Normalized crosstab (if normalize=True)
            - 'row_order': Dendrogram-ordered row labels
            - 'col_order': Column labels
            - 'dendrogram': Dendrogram info for plotting
        """
        # Get subset if specified
        if cell_indices is not None:
            obs_subset = self._adata.obs.iloc[cell_indices]
        else:
            obs_subset = self._adata.obs

        # Create crosstab
        crosstab = pd.crosstab(
            obs_subset[groupby],
            obs_subset[compare_col],
        )

        # Normalize by row if requested
        if normalize:
            row_sums = crosstab.sum(axis=1)
            crosstab_normalized = crosstab.div(row_sums, axis=0)
        else:
            crosstab_normalized = crosstab

        # Get dendrogram for row ordering
        dendro_result = self.get_or_compute_dendrogram(groupby, use_rep=use_rep)
        categories_ordered = dendro_result["categories_ordered"]

        # Filter to categories that exist in crosstab
        row_order = [cat for cat in categories_ordered if cat in crosstab.index]
        # Add any categories not in dendrogram
        for cat in crosstab.index:
            if cat not in row_order:
                row_order.append(cat)

        # Reorder crosstab
        crosstab_ordered = crosstab.loc[row_order, :]
        crosstab_normalized_ordered = crosstab_normalized.loc[row_order, :]

        return {
            "crosstab": crosstab_ordered,
            "crosstab_normalized": crosstab_normalized_ordered,
            "row_order": row_order,
            "col_order": list(crosstab.columns),
            "dendrogram": dendro_result,
            "groupby": groupby,
            "compare_col": compare_col,
        }

    # ==================== Export ====================

    def export_subset(
        self,
        page_id: str,
        output_path: Path,
        include_raw: bool = True,
    ) -> Path:
        """
        Export a subset as a new .h5ad file.

        Args:
            page_id: Page/subset ID to export
            output_path: Path for the output file
            include_raw: Whether to include .raw if present

        Returns:
            Path to the created file
        """
        indices = self._get_subset_indices(page_id)

        # Create a copy (not view) for export
        subset_adata = self._adata[indices].copy()

        # Optionally exclude raw to reduce file size
        if not include_raw:
            subset_adata.raw = None

        # Write to file
        output_path = Path(output_path)
        subset_adata.write_h5ad(output_path)

        return output_path

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
