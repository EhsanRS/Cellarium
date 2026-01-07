# Technical Details

This document explains the technical architecture of Cellarium, focusing on how AnnData objects are processed and transformed into visualizations.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLI Entry                                   │
│                    cellarium serve data.h5ad                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           DataManager                                    │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ Load AnnData    │  │ Precompute       │  │ LRU Expression Cache   │  │
│  │ (full memory)   │→ │ Embeddings       │  │ (100 genes)            │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────────┘  │
│                                                                          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ Subset Manager  │  │ Gene Symbol      │  │ Expression Resolver    │  │
│  │ (numpy indices) │  │ Detection        │  │ (sparse → dense)       │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           SQLite Database                                │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ Pages           │  │ Layouts          │  │ Visualizations         │  │
│  │ (cell_indices   │  │ (panel positions │  │ (plot configs)         │  │
│  │  as pickled     │  │  as JSON)        │  │                        │  │
│  │  numpy arrays)  │  │                  │  │                        │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Plot Rendering                                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ Scatter (WebGL) │  │ Dotplot          │  │ Violin / Heatmap       │  │
│  │ Scattergl trace │  │ Scanpy dotplot   │  │ Plotly traces          │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. AnnData Loading and Memory Management

### Loading Strategy

The `DataManager` class (singleton pattern) loads the entire AnnData object into memory at startup:

```python
# cellarium/data/manager.py:123-142
self._adata = ad.read_h5ad(self.config.data_path)
```

**Design decision:** Full memory loading vs backed mode (`backed='r'`):
- **Full loading** is chosen for medium-sized datasets (50K-500K cells)
- Backed mode would require repeated disk I/O for every plot update
- Memory footprint is acceptable for the target dataset size

### AnnData Structure Reference

```
adata
├── X              : (n_cells × n_genes) - Main expression matrix (usually normalized/log-transformed)
├── raw            : AnnData - Raw counts before filtering (more genes, original counts)
│   ├── X         : (n_cells × raw_n_genes) - Raw expression matrix
│   └── var       : Gene metadata for raw data
├── obs            : DataFrame - Cell metadata (cell_type, batch, n_counts, etc.)
├── var            : DataFrame - Gene metadata (gene_symbols, highly_variable, etc.)
├── obsm           : Dict - Embeddings (X_umap, X_pca, etc.)
├── layers         : Dict - Alternative expression matrices (counts, spliced, unspliced)
└── uns            : Dict - Unstructured data (colors, neighbors, etc.)
```

### Root Subset Initialization

After loading, a "root" subset is created representing all cells:

```python
# cellarium/data/manager.py:132-137
self._subsets["root"] = SubsetInfo(
    page_id="root",
    cell_indices=np.arange(self._adata.n_obs),  # [0, 1, 2, ..., n_cells-1]
    parent_page_id=None,
    n_cells=self._adata.n_obs,
)
```

---

## 2. Embedding Precomputation

### Why Precompute?

Accessing `adata.obsm['X_umap']` repeatedly through AnnData is slower than direct numpy array access. Precomputation converts embeddings to contiguous memory layouts.

### Process

```python
# cellarium/data/manager.py:144-165
def _precompute_embeddings(self) -> None:
    for key in self._adata.obsm.keys():  # e.g., 'X_umap', 'X_pca', 'X_tsne'
        # np.ascontiguousarray ensures memory-efficient row-major layout
        coords = np.ascontiguousarray(self._adata.obsm[key])

        # Generate dimension names: "X_umap" → ["UMAP1", "UMAP2"]
        base_name = key.replace("X_", "").upper()
        dim_names = [f"{base_name}{i+1}" for i in range(coords.shape[1])]

        self._embeddings[key] = EmbeddingData(
            name=key,
            coordinates=coords,  # Shape: (n_cells, n_dims)
            dimensions=dim_names,
        )
```

### Data Structure

```python
@dataclass
class EmbeddingData:
    name: str                    # "X_umap"
    coordinates: np.ndarray      # Shape: (n_cells, n_dims), contiguous memory
    dimensions: List[str]        # ["UMAP1", "UMAP2"]
```

### Retrieval with Subsetting

```python
# cellarium/data/manager.py:426-455
def get_embedding_coords(self, embedding_key: str, page_id: str = "root", dims: Tuple[int, int] = (0, 1)):
    coords = self._embeddings[embedding_key].coordinates

    # Apply subset mask if not root
    if page_id != "root":
        subset = self._subsets.get(page_id)
        coords = coords[subset.cell_indices]  # Fancy indexing

    return coords[:, dims[0]], coords[:, dims[1]]  # Return x, y arrays
```

---

## 3. Expression Data Caching

### LRU Cache Implementation

```python
# cellarium/data/manager.py:54-79
class LRUCache:
    def __init__(self, maxsize: int = 100):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache.move_to_end(key)  # Mark as recently used
            return self._cache[key]
        return None

    def set(self, key: str, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)  # Evict oldest
            self._cache[key] = value
```

**Cache key format:** `"{gene_name}:{layer}:{raw|X}"`
- Example: `"CD3D:X:X"` or `"ENSG00000167286:counts:raw"`

### Expression Retrieval Pipeline

```python
# cellarium/data/manager.py:502-565
def get_gene_expression(self, gene: str, page_id: str = "root", layer: Optional[str] = None,
                        use_raw: bool = False, gene_symbols_column: Optional[str] = None):

    # 1. Resolve gene name (symbol → var_name mapping)
    resolved_gene = self.resolve_gene_name(gene, gene_symbols_column, use_raw)

    # 2. Build cache key
    cache_key = f"{resolved_gene}:{layer or 'X'}:{'raw' if use_raw else 'X'}"

    # 3. Check cache
    cached = self._expression_cache.get(cache_key)
    if cached is None:
        # 4. Select data source (raw vs main, layer vs X)
        if use_raw and self._adata.raw is not None:
            data = self._adata.raw.X
        elif layer and layer in self._adata.layers:
            data = self._adata.layers[layer]
        else:
            data = self._adata.X

        # 5. Get gene index and extract column
        gene_idx = adata_source.var_names.get_loc(resolved_gene)
        expr = data[:, gene_idx]

        # 6. Handle sparse matrices (CSR/CSC from scipy)
        if hasattr(expr, "toarray"):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()

        # 7. Cache the full vector (not subset)
        self._expression_cache.set(cache_key, expr)
        cached = expr

    # 8. Apply subset AFTER caching (maximizes cache reuse)
    indices = self._get_subset_indices(page_id)
    return cached[indices]
```

**Key insight:** The cache stores **full** gene expression vectors, not subsets. Subsetting happens at retrieval time. This maximizes cache reuse across different page views.

---

## 4. Cell Subset Management

### Subset Data Structure

```python
@dataclass
class SubsetInfo:
    page_id: str                        # "page_abc12345"
    cell_indices: np.ndarray            # Integer indices into parent (or root)
    parent_page_id: Optional[str]       # "root" or parent page ID
    n_cells: int
```

### Creating Subsets from Selections

```python
# cellarium/data/manager.py:632-669
def create_subset(self, page_id: str, cell_indices: np.ndarray, parent_page_id: str = "root"):
    """
    IMPORTANT: cell_indices are relative to the PARENT subset, not absolute.
    """
    # Convert relative → absolute indices
    if parent_page_id != "root":
        parent_subset = self._subsets.get(parent_page_id)
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
```

### Index Resolution

```python
# cellarium/data/manager.py:692-711
def _get_subset_indices(self, page_id_or_indices):
    """Flexible index resolution: accepts page_id string, list, or None."""
    if page_id_or_indices is None or page_id_or_indices == "root":
        return np.arange(self._adata.n_obs)  # All cells

    if isinstance(page_id_or_indices, (list, np.ndarray)):
        return np.asarray(page_id_or_indices)  # Direct indices

    # Page ID lookup
    subset = self._subsets.get(page_id_or_indices)
    return subset.cell_indices
```

### Subset Views (Not Copies)

For AnnData operations, subsets return **views** to avoid memory duplication:

```python
# cellarium/data/manager.py:671-679
def get_subset_adata(self, page_id: str) -> ad.AnnData:
    """Returns a VIEW, not a copy. Do not modify!"""
    indices = self._get_subset_indices(page_id)
    return self._adata[indices]  # AnnData fancy indexing returns view
```

---

## 5. Gene Symbol Detection and Resolution

### The Problem

AnnData objects may use different gene identifiers:
- **var_names as Ensembl IDs:** `ENSG00000167286`, `ENSMUSG00000026072`
- **var_names as symbols:** `CD3D`, `Cd3d`
- **Symbols in a var column:** `adata.var['gene_symbol']`

### Detection Algorithm

```python
# cellarium/data/manager.py:232-251
def _looks_like_ensembl_id(self, value: str) -> bool:
    """Detect Ensembl-style identifiers."""
    return value.startswith(('ENS', 'ERCC', 'LINC')) or \
           (len(value) > 10 and value[:4].isalpha() and value[4:].replace('.', '').isdigit())

def _looks_like_gene_symbol(self, value: str) -> bool:
    """Detect gene symbols (short alphanumeric names)."""
    if self._looks_like_ensembl_id(value):
        return False
    if len(value) > 20:
        return False
    return bool(value) and not value.isdigit()
```

### var_names Type Detection

```python
# cellarium/data/manager.py:252-271
def detect_var_names_type(self) -> str:
    """Sample var_names to determine if they're symbols or identifiers."""
    sample_size = min(100, len(self._adata.var_names))
    sample_names = self._adata.var_names[:sample_size]

    ensembl_count = sum(1 for name in sample_names if self._looks_like_ensembl_id(name))

    if ensembl_count > sample_size * 0.8:
        return 'ensembl'
    elif ensembl_count < sample_size * 0.2:
        return 'symbols'
    return 'unknown'
```

### Auto-Detecting Symbol Column

```python
# cellarium/data/manager.py:273-323
def detect_gene_symbols_column(self) -> Optional[str]:
    """Find the best column containing gene symbols."""
    if self.detect_var_names_type() == 'symbols':
        return None  # var_names are already symbols

    candidate_cols = self.get_gene_symbol_column_options()  # String-type columns only

    # Try common names first
    common_names = ['gene_symbol', 'gene_symbols', 'symbol', 'gene_name',
                    'external_gene_name', 'gene_short_name', 'Symbol', 'GeneName']

    for name in common_names:
        if name in candidate_cols:
            return name

    # Score columns by symbol-likeness
    for col in candidate_cols:
        values = self._adata.var[col].dropna().astype(str)
        sample_values = values.iloc[:min(100, len(values))]
        symbol_count = sum(1 for v in sample_values if self._looks_like_gene_symbol(v))
        score = symbol_count / len(sample_values)
        if score > 0.5:
            return col

    return None
```

### Gene Name Resolution Pipeline

```python
# cellarium/data/manager.py:397-422
def resolve_gene_name(self, gene: str, gene_symbols_column: Optional[str], use_raw: bool):
    """Convert display name (symbol) → var_name for indexing."""
    adata = self._adata.raw if use_raw and self._adata.raw else self._adata

    # Direct match in var_names
    if gene in adata.var_names:
        return gene

    # Symbol → var_name mapping via column
    if gene_symbols_column and gene_symbols_column in adata.var.columns:
        mask = adata.var[gene_symbols_column] == gene
        if mask.any():
            return adata.var_names[mask][0]

    raise ValueError(f"Gene not found: {gene}")
```

---

## 6. Data Flow to Plot Rendering

### Scatter Plot (UMAP/PCA)

```python
# Data flow for scatter plot:
#
# 1. Get embedding coordinates
coords_x, coords_y = data_manager.get_embedding_coords("X_umap", page_id="page_123")
# Returns: (numpy array of x, numpy array of y) for subset cells

# 2. Get coloring data (categorical or continuous)
if color_by_gene:
    color_values = data_manager.get_gene_expression("CD3D", page_id="page_123", use_raw=True)
else:
    color_values = data_manager.get_obs_column("cell_type", page_id="page_123")

# 3. Build DataFrame for Plotly
df = pd.DataFrame({
    "x": coords_x,
    "y": coords_y,
    "cell_index": subset_indices,  # For selection tracking
    "color": color_values,
})

# 4. Create Scattergl trace (WebGL for 500K+ cells)
trace = go.Scattergl(
    x=df["x"],
    y=df["y"],
    mode="markers",
    marker=dict(
        color=df["color"],
        colorscale="viridis",
        size=point_size,
    ),
    customdata=df["cell_index"],  # Used for lasso/box selection
)
```

### Dotplot

```python
# Data flow for dotplot:
#
# 1. Get expression for multiple genes
gene_list = ["CD3D", "CD4", "CD8A", "MS4A1"]
expr_df = data_manager.get_genes_expression(
    genes=gene_list,
    page_id="page_123",
    use_raw=True,
    gene_symbols_column="gene_symbol"
)
# Returns: DataFrame with genes as columns, cells as rows

# 2. Get grouping variable
groups = data_manager.get_obs_column("cell_type", page_id="page_123")

# 3. Compute per-group statistics
for group in groups.unique():
    mask = groups == group
    group_expr = expr_df[mask]

    # Fraction expressing (> threshold)
    fraction = (group_expr > expression_cutoff).mean()

    # Mean expression (optionally only in expressing cells)
    if mean_only_expressed:
        mean_expr = group_expr[group_expr > 0].mean()
    else:
        mean_expr = group_expr.mean()

# 4. Build dot plot (dot size = fraction, color = mean expression)
```

### Violin Plot

```python
# Data flow for violin plot:
#
# 1. Get expression for selected genes
genes = ["CD3D", "CD4"]
expr_df = data_manager.get_genes_expression(genes, page_id="page_123", use_raw=True)

# 2. Get grouping variable
groups = data_manager.get_obs_column("cell_type", page_id="page_123")

# 3. Build violin traces per gene per group
for gene in genes:
    for group in groups.unique():
        mask = groups == group
        values = expr_df[gene][mask]

        trace = go.Violin(
            y=values,
            name=f"{gene} - {group}",
            box_visible=True,
            points="all" if show_stripplot else False,
        )
```

---

## 7. SQLite Persistence Architecture

### Database Initialization

```python
# cellarium/db/schema.py:111-143
def init_database(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # WAL mode: better concurrent read performance
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript(SCHEMA_SQL)
    return conn
```

### Schema Overview

```sql
-- Pages: Data subsets with cell indices
CREATE TABLE pages (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_page_id TEXT,
    cell_indices BLOB,          -- Pickled numpy array
    n_cells INTEGER NOT NULL,
    FOREIGN KEY (parent_page_id) REFERENCES pages(id) ON DELETE CASCADE
);

-- Layouts: Panel arrangements (positions, sizes)
CREATE TABLE layouts (
    id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL,
    layout_json TEXT NOT NULL,  -- JSON: {lg: [...], md: [...], sm: [...]}
    is_active BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
);

-- Visualizations: Plot configurations
CREATE TABLE visualizations (
    id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL,
    viz_type TEXT NOT NULL,     -- 'scatter', 'dotplot', 'violin', 'heatmap'
    config_json TEXT NOT NULL,  -- Full plot configuration
    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
);
```

### Cell Indices Storage

Cell indices are stored as pickled numpy arrays (BLOB):

```python
# cellarium/db/repository.py:97-109
def create(self, name: str, cell_indices: np.ndarray, parent_page_id: str = "root"):
    indices_blob = pickle.dumps(cell_indices)  # Serialize numpy array

    conn.execute(
        "INSERT INTO pages (id, name, parent_page_id, cell_indices, n_cells) VALUES (?, ?, ?, ?, ?)",
        (page_id, name, parent_page_id, indices_blob, len(cell_indices)),
    )

# Retrieval:
def _row_to_page(self, row: sqlite3.Row) -> Page:
    return Page(
        cell_indices=pickle.loads(row["cell_indices"]) if row["cell_indices"] else None,
        ...
    )
```

### Repository Pattern

Each entity type has a dedicated repository class:

- `PageRepository`: CRUD for data subsets
- `SelectionRepository`: Cell selections (lasso/box)
- `LayoutRepository`: Dashboard panel arrangements
- `VisualizationRepository`: Plot configurations
- `GeneListRepository`: Saved gene sets

---

## 8. Performance Optimizations

| Component | Strategy | Impact |
|-----------|----------|--------|
| Scatter plots | `go.Scattergl` (WebGL) | Renders 500K+ cells |
| Embeddings | Contiguous numpy arrays | Fast array slicing |
| Expression | LRU cache (100 genes) | Avoids repeated sparse→dense |
| Subsets | Views not copies | Minimal memory overhead |
| Selection | `customdata` attribute | O(1) cell index lookup |
| Database | WAL mode | Concurrent reads |

### Sparse Matrix Handling

AnnData stores expression matrices as sparse (CSR/CSC) for memory efficiency:

```python
# Sparse extraction and densification:
expr = data[:, gene_idx]          # Returns sparse column
if hasattr(expr, "toarray"):
    expr = expr.toarray().flatten()  # Convert to dense 1D array
```

### vmax Percentile Calculation

For gene expression coloring, vmax can be set to a percentile:

```python
if vmax_type == "percentile":
    # Only consider non-zero values for percentile
    positive_values = color_values[color_values > 0]
    if len(positive_values) > 0:
        vmax = np.percentile(positive_values, vmax_value)  # e.g., 99.9th percentile
```

---

## 9. Data Export

### Subset Export Pipeline

```python
# cellarium/data/manager.py:743-773
def export_subset(self, page_id: str, output_path: Path, include_raw: bool = True):
    indices = self._get_subset_indices(page_id)

    # Create COPY (not view) for independent file
    subset_adata = self._adata[indices].copy()

    if not include_raw:
        subset_adata.raw = None  # Reduce file size

    subset_adata.write_h5ad(output_path)
```

**Note:** Export creates a **copy** because views cannot be written to file independently.
