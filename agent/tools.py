def lookup_stock_price(stock_symbol: str) -> str:
    """Useful for looking up a stock price."""
    print(f"Looking up stock price for {stock_symbol}")
    return f"Symbol {stock_symbol} is currently trading at $100.00"


def search_for_stock_symbol(symbol: str) -> str:
    """Useful for searching for a stock symbol given a free-form company name."""
    print("Searching for stock symbol")
    return symbol.upper()


def add(x: int, y: int) -> int:
    """Useful function to add two numbers."""
    return x + y


def multiply(x: int, y: int) -> int:
    """Useful function to multiply two numbers."""
    return x * y


def web_search(query: str) -> str:
    """Useful function to search the web and returns the result."""
    if "weather" in query:
        return "The weather in New York is 75°F, clear skies."
    return "No relevant information found."


def social_media_post(content: str) -> str:
    """Useful function to post on social media."""
    print(f"Posting to social media: {content}")
    return "Post successful!"
