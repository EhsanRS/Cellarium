<p align="center">
  <img src="https://img.shields.io/badge/🔬-Cellarium-blue?style=for-the-badge&labelColor=1a1a2e" alt="Cellarium"/>
</p>

<h1 align="center">Cellarium</h1>

<p align="center">
  <strong>Interactive single-cell data exploration in your browser</strong>
</p>

<p align="center">
  <a href="https://github.com/EhsanRS/Cellarium/actions/workflows/ci.yml">
    <img src="https://github.com/EhsanRS/Cellarium/actions/workflows/ci.yml/badge.svg" alt="CI Status"/>
  </a>
  <a href="https://pypi.org/project/Cellarium/">
    <img src="https://img.shields.io/pypi/v/cellarium?color=blue" alt="PyPI Version"/>
  </a>
  <a href="https://pypi.org/project/Cellarium/">
    <img src="https://img.shields.io/pypi/pyversions/cellarium" alt="Python Versions"/>
  </a>
  <a href="https://github.com/EhsanRS/Cellarium/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
  </a>
  <a href="https://github.com/EhsanRS/Cellarium/stargazers">
    <img src="https://img.shields.io/github/stars/EhsanRS/cellarium?style=social" alt="GitHub Stars"/>
  </a>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-cli-reference">CLI Reference</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## Overview

**Cellarium** is a lightweight, local web application for exploring single-cell RNA-seq data stored in AnnData (`.h5ad`) format. Built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/), it provides interactive visualizations that can handle **500,000+ cells** smoothly using WebGL rendering.

No cloud uploads. No installations headaches. Just point it at your `.h5ad` file and start exploring.

```bash
pip install cellarium
cellarium serve your_data.h5ad
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **WebGL Scatter Plots** | Interactive UMAP/PCA/t-SNE plots that handle 500K+ cells smoothly |
| **Expression Plots** | Dot plots, violin plots, and heatmaps for gene expression analysis |
| **Cell Subsetting** | Lasso/box select cells to create new analysis pages |
| **Multi-Panel Dashboard** | Drag, resize, and arrange multiple plots side by side |
| **Dark/Light Themes** | Easy on the eyes, day or night |
| **Persistent Sessions** | Your pages and layouts survive browser refreshes |
| **Data Export** | Export cell subsets as new `.h5ad` files |
| **Variable Filtering** | Create subsets by metadata (e.g., cluster 1, 5, 10) or gene expression (e.g., CD68 > 1) |

---

## 🚀 Quick Start

### Option 1: pip install (Recommended)

```bash
# Create a virtual environment (recommended)
python -m venv cellarium-env
source cellarium-env/bin/activate  # On Windows: cellarium-env\Scripts\activate

# Install Cellarium
pip install cellarium

# Run with your data
cellarium serve your_data.h5ad
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/EhsanRS/Cellarium.git
cd cellarium

# Create environment and install
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

# Run
cellarium serve your_data.h5ad
```

### Option 3: Conda

```bash
# Create conda environment
conda create -n cellarium python=3.11
conda activate cellarium

# Install from source
git clone https://github.com/EhsanRS/Cellarium.git
cd cellarium
pip install -e .

# Run
cellarium serve your_data.h5ad
```

After running, open your browser to **http://127.0.0.1:8050** (opens automatically).

---

## 📖 Usage

### Selecting Cells

- Use the **lasso tool** or **box select** on any scatter plot
- Selected cells are highlighted across all plots
- Press **Escape** to clear selection

### Creating Subsets

**From Selection:**
1. Select cells on a scatter plot
2. Click **"Create Page"** in the bottom bar
3. Name your subset → A new page is created

**From Filters:**
1. Click the **+** button in the sidebar
2. Choose **By Variable** (e.g., leiden clusters 1, 5, 10)
3. Or choose **By Gene Expression** (e.g., CD68 > 1)

### Managing Plots

- Click **"Add Panel"** to add new visualizations
- Choose from: UMAP, PCA, t-SNE, Dot Plot, Violin, Heatmap, Crosstab
- **Drag** panels to rearrange
- **Resize** by dragging corners
- Click the **gear icon** to configure each plot

### Saving & Exporting

- **Save Layout**: Download your dashboard configuration as JSON
- **Load Layout**: Restore a saved configuration
- **Export**: Save selected cells as a new `.h5ad` file

---

## 💻 CLI Reference

```bash
cellarium serve DATA_PATH [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host`, `-h` | `127.0.0.1` | Host address to bind |
| `--port`, `-p` | `8050` | Port number (1024-65535) |
| `--db`, `-d` | `./cellarium.db` | SQLite database path |
| `--debug` | `False` | Enable debug mode with hot reload |
| `--no-browser` | `False` | Don't auto-open browser |
| `--version`, `-v` | - | Show version and exit |

### Examples

```bash
# Basic usage
cellarium serve pbmc3k.h5ad

# Custom port
cellarium serve data.h5ad --port 8080

# Allow network access
cellarium serve data.h5ad --host 0.0.0.0

# Development mode
cellarium serve data.h5ad --debug

# Custom database location
cellarium serve data.h5ad --db ~/analysis/session.db

# Quick file info without starting server
cellarium info data.h5ad
```

---

## 🔧 Troubleshooting

<details>
<summary><strong>"Command not found: cellarium"</strong></summary>

Make sure you:
1. Activated your virtual environment
2. Installed the package: `pip install -e .` or `pip install cellarium`

</details>

<details>
<summary><strong>"Module not found" errors</strong></summary>

Try reinstalling:
```bash
pip install -e . --force-reinstall
```

</details>

<details>
<summary><strong>Slow performance with large datasets</strong></summary>

- Cellarium uses WebGL rendering, which should handle 500K+ cells
- Use Chrome or Firefox for best performance
- Ensure your browser supports WebGL: [WebGL Test](https://get.webgl.org/)

</details>

<details>
<summary><strong>Port already in use</strong></summary>

Use a different port:
```bash
cellarium serve data.h5ad --port 8051
```

</details>

---

## 📋 Requirements

- **Python**: 3.10 or higher
- **Browser**: Chrome, Firefox, Safari, or Edge (with WebGL support)
- **Data**: AnnData (`.h5ad`) file with embeddings (e.g., `X_umap`, `X_pca`)

### Dependencies

Core dependencies are automatically installed:
- `dash` >= 2.14.0
- `dash-mantine-components` >= 0.14.0
- `plotly` >= 5.18.0
- `anndata` >= 0.10.0
- `scanpy` >= 1.9.0
- `pandas` >= 2.0.0
- `numpy` >= 1.24.0

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# Clone and install in development mode
git clone https://github.com/EhsanRS/cellarium.git
cd cellarium
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy cellarium
```

### Development Guidelines

- Follow existing code style (enforced by `ruff`)
- Add tests for new features
- Update documentation as needed
- Use conventional commit messages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with amazing open-source tools:
- [Dash](https://dash.plotly.com/) - Web framework
- [Plotly](https://plotly.com/) - Interactive visualizations
- [AnnData](https://anndata.readthedocs.io/) - Data structures
- [Scanpy](https://scanpy.readthedocs.io/) - Single-cell analysis
- [Dash Mantine Components](https://www.dash-mantine-components.com/) - UI components

---

<p align="center">
  Made with ❤️ for the single-cell community
</p>
