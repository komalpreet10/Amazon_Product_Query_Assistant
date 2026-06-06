"""
citations.py
------------
Source citation helpers for product-grounded RAG answers.
"""


def build_citations(products: list[dict]) -> list[dict]:
    """
    Converts retrieved products into compact citation metadata.
    """
    citations = []
    for index, product in enumerate(products, start=1):
        citations.append(
            {
                "id": index,
                "parent_asin": product.get("parent_asin"),
                "title": product.get("title", "N/A"),
                "store": product.get("store"),
                "price": product.get("price"),
                "average_rating": product.get("average_rating"),
                "rating_number": product.get("rating_number"),
            }
        )
    return citations
