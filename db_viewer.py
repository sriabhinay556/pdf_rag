"""
ChromaDB Data Viewer — inspect every collection stored in ``chroma_store/``.

Run with:  ``streamlit run db_viewer.py``

Tabs: Documents (page text), Metadata, Embedding statistics & heatmap, Raw data.
"""

import chromadb
import numpy as np
import pandas as pd
import streamlit as st

import config

# ── Page setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="ChromaDB Viewer", page_icon="🗄️", layout="wide")
st.title("ChromaDB Data Viewer")


# ── Helpers ───────────────────────────────────────────────────────────

@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def load_collection_data(collection_name: str) -> dict:
    """Fetch everything from a ChromaDB collection."""
    client = get_client()
    collection = client.get_collection(name=collection_name)
    return collection.get(include=["documents", "metadatas", "embeddings"])


# ── Sidebar: collection picker ────────────────────────────────────────

client = get_client()
collections = client.list_collections()

if not collections:
    st.warning("No collections found in ChromaDB. Ingest a PDF first.")
    st.stop()

with st.sidebar:
    st.header("Collections")

    collection_names = sorted([c.name for c in collections])
    selected = st.radio(
        "Select a collection",
        collection_names,
        index=0,
    )

    st.divider()
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


# ── Load data for selected collection ─────────────────────────────────

data = load_collection_data(selected)

ids = data.get("ids", [])
documents = data.get("documents", [])
metadatas = data.get("metadatas", [])
embeddings = data.get("embeddings", [])

total_docs = len(ids)

# ── Overview metrics ──────────────────────────────────────────────────

st.subheader(f"Collection: `{selected}`")

col1, col2, col3 = st.columns(3)
col1.metric("Documents", total_docs)
if embeddings is not None and len(embeddings) > 0:
    col2.metric("Embedding Dimensions", len(embeddings[0]))
else:
    col2.metric("Embedding Dimensions", "N/A")
col3.metric(
    "Avg Document Length",
    f"{np.mean([len(d) for d in documents]):.0f} chars" if documents else "N/A",
)

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────

tab_docs, tab_meta, tab_embed, tab_raw = st.tabs(
    ["📄 Documents", "🏷️ Metadata", "📐 Embeddings", "🔍 Raw Data"]
)

# ── Tab 1: Documents ──────────────────────────────────────────────────

with tab_docs:
    st.markdown("**Structured text stored as `page_content` in each document.**")

    # Build a dataframe
    doc_rows = []
    for doc_id, doc_text, meta in zip(ids, documents, metadatas):
        page_num = meta.get("page_number", "—")
        title = meta.get("title", "—")
        doc_rows.append(
            {
                "ID": doc_id,
                "Page": page_num,
                "Title": title,
                "Content (preview)": doc_text,
                "Length": len(doc_text),
            }
        )

    df_docs = pd.DataFrame(doc_rows)
    if "Page" in df_docs.columns:
        df_docs = df_docs.sort_values("Page").reset_index(drop=True)

    st.dataframe(df_docs, use_container_width=True, height=500)

    # Expandable full-text viewer
    st.markdown("---")
    st.markdown("**Full document text** — click to expand:")
    for doc_id, doc_text, meta in sorted(
        zip(ids, documents, metadatas),
        key=lambda x: x[2].get("page_number", 0),
    ):
        page = meta.get("page_number", "?")
        title = meta.get("title", doc_id)
        with st.expander(f"Page {page} — {title}"):
            st.text(doc_text)

# ── Tab 2: Metadata ──────────────────────────────────────────────────

with tab_meta:
    st.markdown("**Metadata stored alongside each document.**")

    # Flatten metadata dicts into a dataframe
    meta_rows = []
    for doc_id, meta in zip(ids, metadatas):
        row = {"ID": doc_id}
        row.update(meta)
        meta_rows.append(row)

    df_meta = pd.DataFrame(meta_rows)

    # Truncate very long columns for display (e.g. summary)
    display_df = df_meta.copy()
    for col in display_df.columns:
        if display_df[col].dtype == object:
            display_df[col] = display_df[col].apply(
                lambda v: (str(v)[:200] + "…") if isinstance(v, str) and len(v) > 200 else v
            )

    if "page_number" in display_df.columns:
        display_df = display_df.sort_values("page_number").reset_index(drop=True)

    st.dataframe(display_df, use_container_width=True, height=500)

    # Column-level stats
    st.markdown("---")
    st.markdown("**Metadata fields present:**")
    if meta_rows:
        all_keys = sorted({k for row in meta_rows for k in row if k != "ID"})
        for key in all_keys:
            values = [row.get(key) for row in meta_rows if row.get(key)]
            st.markdown(f"- `{key}` — **{len(values)}** / {total_docs} documents have this field")

# ── Tab 3: Embeddings ────────────────────────────────────────────────

with tab_embed:
    if embeddings is None or len(embeddings) == 0:
        st.info("No embeddings stored or not included in this collection.")
    else:
        emb_array = np.array(embeddings)
        st.markdown(f"**Shape:** {emb_array.shape[0]} vectors × {emb_array.shape[1]} dimensions")

        # Summary statistics
        st.markdown("**Embedding statistics:**")
        stats = {
            "Min": float(emb_array.min()),
            "Max": float(emb_array.max()),
            "Mean": float(emb_array.mean()),
            "Std": float(emb_array.std()),
        }
        stat_cols = st.columns(4)
        for i, (label, val) in enumerate(stats.items()):
            stat_cols[i].metric(label, f"{val:.6f}")

        st.divider()

        # Per-vector norms
        st.markdown("**Per-document vector norms (L2):**")
        norms = np.linalg.norm(emb_array, axis=1)
        norm_df = pd.DataFrame(
            {
                "ID": ids,
                "Page": [m.get("page_number", "?") for m in metadatas],
                "L2 Norm": norms,
            }
        )
        if "Page" in norm_df.columns:
            norm_df = norm_df.sort_values("Page").reset_index(drop=True)
        st.dataframe(norm_df, use_container_width=True)

        # Heatmap of first N dimensions
        st.divider()
        max_dims = min(50, emb_array.shape[1])
        st.markdown(f"**Embedding heatmap** (first {max_dims} dims):")
        heatmap_df = pd.DataFrame(
            emb_array[:, :max_dims],
            index=[f"Page {m.get('page_number', i)}" for i, m in enumerate(metadatas)],
            columns=[f"d{j}" for j in range(max_dims)],
        )
        st.dataframe(
            heatmap_df.style.background_gradient(cmap="RdBu_r", axis=None),
            use_container_width=True,
            height=400,
        )

# ── Tab 4: Raw Data ──────────────────────────────────────────────────

with tab_raw:
    st.markdown("**Complete raw data from `collection.get()`.**")

    st.markdown(f"Showing all {total_docs} entries.")
    for i, (doc_id, doc_text, meta) in enumerate(zip(ids, documents, metadatas)):
        with st.expander(f"Entry {i}: {doc_id}"):
            st.markdown("**ID:**")
            st.code(doc_id)
            st.markdown("**Metadata:**")
            st.json(meta)
            st.markdown("**Document text:**")
            st.text(doc_text)
            if embeddings is not None and i < len(embeddings):
                st.markdown(f"**Embedding:** {len(embeddings[i])} dimensions")
                st.code(str(embeddings[i][:20]) + " ...")
