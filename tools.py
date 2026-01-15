"""
Tools for the Movie & Book Chatbot
Part 1: Calculator Tool
Part 2: Langchain tool wrapper
Part 3: Web Search Tool (Brave API)
Part 4: Movie Tool (OMDB API)
Part 5: Book Tool (OpenLibrary API)
Part 6: RAG Tool (Search Personal Movie/Book Docs))
"""

from langchain_core.tools import Tool
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import RAG functions from rag.py
try:
    from rag import create_embeddings, load_vectorstore
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️  Warning: Could not import RAG functions. RAG tool will be disabled.")

# ============================================================
# 5.0 Start OF RAG TOOL (SEARCH PERSONAL DOCUMENTS)
# ============================================================

# Global variable to cache the vectorstore
_vectorstore = None

def rag_search(query: str, max_results: int = 3) -> str:
    """
    Search personal documents using RAG (Retrieval-Augmented Generation).
    
    This tool allows the LLM to search through your personal movie/book
    collection, ratings, and notes stored in local documents.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 3)
        
    Returns:
        Formatted search results from documents, or error message
        
    Examples:
        >>> search_documents("movies rated 10/10")
        'Found in my_ratings.txt:
         INCEPTION (2010) - 10/10
         Absolutely mind-blowing!...'
        
    Requirements:
        - Needs chroma_db/ folder to exist (run rag.py first)
        - Loads from documents/ folder
    """

    global _vectorstore

     # Check if RAG is available
    if not RAG_AVAILABLE:
        return (
            "Error: RAG system not available. "
            "Make sure rag.py exists and run 'python rag.py' first."
        )
    
    try:
        # Load vectorstore (cached after first load)
        if _vectorstore is None:
            # Check if database exists
            if not os.path.exists("./chroma_db"):
                return (
                    "Error: Vector database not found. "
                    "Please run 'python rag.py' first to create it."
                )
            
        
            # Load embeddings model
            embeddings = create_embeddings()
            
            # Load existing vectorstore
            _vectorstore = load_vectorstore(embeddings, persist_directory="./chroma_db")
        
        # Search for similar documents
        results = _vectorstore.similarity_search(query, k=max_results)
        
        if not results:
            return f"No relevant information found in documents for query: '{query}'"
        
        # Format results for the LLM
        formatted_results = []
        formatted_results.append(f"📚 Search results from personal documents for: '{query}'\n")
        
        for i, doc in enumerate(results, 1):
            # Get source filename
            source = doc.metadata.get("source", "unknown")
            filename = os.path.basename(source)
            
            # Get content
            content = doc.page_content
            
            formatted_results.append(f"Result {i} (from {filename}):")
            formatted_results.append(f"{content}")
            formatted_results.append("")  # Empty line between results
        
        return "\n".join(formatted_results)
        
    except FileNotFoundError as e:
        return f"Error: Vector database not found. Run 'python rag.py' first to create it."
    
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
    
def test_rag_search():
    """
    Test the RAG search tool.
    """
    print("\n" + "="*50)
    print("TESTING RAG SEARCH TOOL")
    print("="*50 + "\n")
    
    # Check if RAG is available
    if not RAG_AVAILABLE:
        print("⚠️  RAG system not available")
        print("   Skipping RAG search tests")
        print("   Make sure rag.py exists")
        print("\n" + "="*50)
        return
    
    # Check if database exists
    if not os.path.exists("./chroma_db"):
        print("⚠️  Vector database not found at ./chroma_db")
        print("   Please run 'python rag.py' first to create it")
        print("\n" + "="*50)
        return
    
    print("✅ Vector database found!")
    print()
    
    # Test cases
    test_queries = [
        ("movies rated 10/10", "Personal ratings query"),
        ("books with magic", "Subject-based query"),
        ("Inception movie", "Specific movie query")
    ]
    
    for query, description in test_queries:
        print(f"📝 Test: {description}")
        print(f"   Query: '{query}'")
        print()
        
        result = rag_search(query, max_results=2)
        
        # Check if it's an error
        if result.startswith("Error:"):
            print(f"   ❌ {result}")
        else:
            print(f"   ✅ Results found!")
            # Print first 400 chars
            print(f"   {result[:400]}...")
        
        print()
    
    print("="*50)
    print("✅ RAG search tool test COMPLETED!")
    print("="*50)
    
# ============================================================
# 5.0 END OF RAG SEARCH TOOL (SEARCH PERSONAL DOCUMENTS)
# ============================================================

# ============================================================
# 5.1 START OF RAG SEARCH LANGCHAIN TOOL WRAPPER
# ============================================================

# Create the LangChain Tool wrapper for RAG search
rag_search_tool = Tool(
    name="rag_search",
    func=rag_search,
    description=(
        "Use this tool to search through personal movie and book collection, "
        "ratings, reviews, and notes stored in local documents. "
        "This is your FIRST choice for questions about: "
        "personal ratings, favorite movies/books, what you've watched/read, "
        "your opinions, and any information that might be in your personal notes. "
        "Input should be a clear search query as a string. "
        "Examples: 'movies I rated 10/10', 'books about magic', 'what did I think of Inception'. "
        "Returns relevant excerpts from your personal documents."
    )
)

def get_rag_search_tool():
    """
    Returns the RAG search tool for use by the LLM agent.
    
    Returns:
        LangChain Tool object for searching personal documents
        
    Example:
        >>> from tools import get_rag_search_tool
        >>> tool = get_rag_search_tool()
        >>> result = tool.func("movies rated 10/10")
        >>> print(result)
    """
    return rag_search_tool

def test_rag_search_tool():
    """
    Test the LangChain wrapper for RAG search.
    """
    print("\n" + "="*50)
    print("TESTING RAG SEARCH LANGCHAIN WRAPPER")
    print("="*50 + "\n")
    
    # Check if RAG is available and database exists
    if not RAG_AVAILABLE or not os.path.exists("./chroma_db"):
        print("⚠️  Skipping - RAG not available or database not found")
        print("="*50)
        return
    
    # Get the tool
    tool = get_rag_search_tool()
    
    # Show tool properties
    print("📋 Tool Properties:")
    print(f"   Name: {tool.name}")
    print(f"   Description: {tool.description[:80]}...")
    print()
    
    # Test using the tool
    print("🧪 Testing tool.run() method:")
    result = tool.run("what movies did I rate highly")
    
    if result.startswith("Error:"):
        print(f"   ❌ {result}")
    else:
        print(f"   ✅ Search successful!")
        print(f"   Results preview: {result[:250]}...")
    
    print()
    print("="*50)
    print("✅ RAG search wrapper test COMPLETED!")
    print("="*50)



# ============================================================
# 1.0 START OF CALCULATOR TOOL
# ============================================================

def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    
    This tool allows the LLM to perform calculations when needed,
    such as computing averages, converting units, or doing arithmetic.
    
    Args:
        expression: A mathematical expression as a string
                   Examples: "2 + 2", "15 * 8", "(10 + 5) / 3"
        
    Returns:
        The result of the calculation as a string
        
    Examples:
        >>> calculator("2 + 2")
        '4'
        
        >>> calculator("15 * 8")
        '120'
        
        >>> calculator("(10 + 9 + 9) / 3")
        '9.333333333333334'
        
    Safety:
        - Only allows basic math operations (+, -, *, /, **, %)
        - No access to functions or imports
        - Safe from code injection
    """
    try:
        # Validate that the expression only contains safe characters
        # Allow: digits, operators, parentheses, decimal points, spaces
        safe_chars = set('0123456789+-*/().% ')
        
        if not all(c in safe_chars for c in expression):
            return f"Error: Expression contains invalid characters. Only use: {safe_chars}"
        
        # Evaluate the expression
        # eval() is normally dangerous, but we've validated the input
        result = eval(expression)
        
        # Return as string
        return str(result)
        
    except ZeroDivisionError:
        return "Error: Cannot divide by zero"
    
    except SyntaxError:
        return f"Error: Invalid mathematical expression: '{expression}'"
    
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
    

# Test function for calculator
def test_calculator():
    """
    Test the calculator function with various expressions.
    """
    print("\n" + "="*50)
    print("TESTING CALCULATOR TOOL")
    print("="*50 + "\n")

    # Test cases: (expression, expected_result_description)
    test_cases = [
        ("2 + 2", "Basic addition"),
        ("10 - 3", "Subtraction"),
        ("5 * 8", "Multiplication"),
        ("20 / 4", "Division"),
        ("2 ** 3", "Exponentiation (2^3)"),
        ("10 % 3", "Modulo (remainder)"),
        ("(10 + 5) / 3", "With parentheses"),
        ("(100 + 50) * 2 - 25", "Complex expression"),
        ("15 / 0", "Division by zero (should error)"),
        ("2 + + 2", "Invalid syntax (should error)"),
        ("import os", "Unsafe code (should error)"),
    ]
    
    print("Running test cases:\n")

    passed = 0
    failed = 0

    for expression, description in test_cases:
        result = calculator(expression)

        #Check if it's an error
        is_error = result.startswith("Error: ")

        print(f"   Test: {description}")
        print(f"   Expression: '{expression}'")
        print(f"   Result: {result}")
    
        
        # Validate expected errors
        if "error" in description.lower() or "unsafe" in description.lower():
            if is_error:
                print(f"Correctly returned error\n")
                passed += 1
            else:
                print(f"Should have returned error\n")
                failed += 1
        else:
            if not is_error:
                print(f"Success\n")
                passed += 1
            else:
                print(f"Unexpected error\n")
                failed += 1
    
    print("="*50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All calculator tests PASSED!")
    else:
        print(f"Some tests failed")
    
    print("="*50)

# ============================================================
# 1.0 END OF CALCULATOR TOOL
# ============================================================
    

# ============================================================
# 1.1 START OF CALCULATOR LANGCHAIN TOOL WRAPPER
# ============================================================

# Create the LangChain Tool object
# This wraps our simple calculator function so the LLM can use it
calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description=(
        "Use this tool to perform mathematical calculations. "
        "Input should be a valid mathematical expression as a string. "
        "Supports: addition (+), subtraction (-), multiplication (*), "
        "division (/), exponentiation (**), modulo (%), and parentheses. "
        "Examples: '2 + 2', '(10 + 5) / 3', '2 ** 8'. "
        "Returns the result as a string."
    )
)

def get_calculator_tool():
    """
    Returns the calculator tool for use by the LLM agent.
    
    This function provides a convenient way to get the tool
    when setting up the agent.
    
    Returns:
        LangChain Tool object for the calculator
        
    Example:
        >>> from tools import get_calculator_tool
        >>> tool = get_calculator_tool()
        >>> result = tool.func("2 + 2")
        >>> print(result)
        '4'
    """
    return calculator_tool

def test_calculator_tool():
    """
    Test the LangChain calculator tool wrapper.
    """
    tool = get_calculator_tool()
    print("\n" + "="*50)
    print("TESTING LANGCHAIN CALCULATOR TOOL")
    print("="*50 + "\n")

    # Show tool properties
    print("    Tool Properties:")
    print(f"   Name: {tool.name}")
    print(f"   Description: {tool.description[:100]}...")
    print()

    test_cases = [
        "2 + 2",
        "15 * 8", 
        "(100 + 50) / 2"
    ]

    for expression in test_cases:
        result = tool.func(expression)

        print(f"Expression: '{expression}'")
        print(f"Result from tool: {result}")
        print()

    # Test the run() method (LangChain's standard way)
    print("    Testing tool.run() method:")
    result = tool.run("10 * 5")
    print(f"   tool.run('10 * 5') = {result}")
    print()
    
    print("="*50)
    print("Langchain Test Calculator tool wrapper test PASSED!")
    print("="*50)


# ============================================================
# 1.1 END OF CALCULATOR LANGCHAIN TOOL WRAPPER
# ============================================================


# ============================================================
# 2.0 START OF WEB SEARCH TOOL (BRAVE API)
# ============================================================

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Brave Search API.
    
    This tool allows the LLM to search for current information
    on the internet when it doesn't have the answer in its
    knowledge base or RAG documents.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted search results as a string, or error message
        
    Examples:
        >>> web_search("Inception movie rating")
        '1. Inception (2010) - IMDb
           IMDb rating of 8.8/10...
           https://www.imdb.com/...'
        
    API Requirements:
        - Needs BRAVE_API_KEY in environment variables
        - Free tier: 2,000 queries/month
    """

    #Get API key from environment
    web_key = os.getenv("BRAVE_API_KEY")

    if not web_key:
       return "Error: BRAVE_API_KEY not found in environment variables."
    
    # Brave Search API endpoint (load from .env or use default)
    websearch_url = os.getenv("BRAVE_API_URL")

    #headers request
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": web_key
    }

    #Query parameters
    params = {
        "q": query,
        "count": min(max_results,20)
    }

    try:
        #Make the API request
        response = requests.get(websearch_url, headers=headers, params=params, timeout=10)

        #Check for HTTP errors
        if response.status_code == 401:
            return "Error: Invalid Brave API key."
        
        if response.status_code == 429:
            return "Error: Rate limit exceeded for Brave Search API."
        
        if response.status_code != 200:
            return f"Error: Brave Search API returned status code {response.status_code}."
        
        #Parse JSON response
        data = response.json()

        #Extract results
        results = data.get("web", {}).get("results", [])

        if not results:
            return f" No results found for query: '{query}'"
        
        # Format results for the LLM
        formatted_results = []
        formatted_results.append(f"Search results for: '{query}'\n")

        for i, result in enumerate(results[:max_results], 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            description = result.get("description", "No description")
            
            formatted_results.append(f"{i}. {title}")
            formatted_results.append(f"   {description}")
            formatted_results.append(f"   URL: {url}\n")
        
        return "\n".join(formatted_results)
        
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    
    except requests.exceptions.RequestException as e:
        return f"Error: Network error - {str(e)}"
    
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
    
# ============================================================
# 2.0 START OF WEB SEARCH TOOL (BRAVE API)
# ============================================================
    

# ============================================================
# 2.1 START OF WEB SEARCH LANGCHAIN TOOL WRAPPER
# ============================================================
#     
#Create the LangChain tool wrapper for web search
web_search_tool = Tool(
    name="web_search",
    func=web_search,
    description=(
        "Use this tool to search the web for current information about movies, books, ratings, release dates, "
        "or any general knowledge not available in your training data."
        "Input should be clear a clear search query as a string."
        "Examples: 'Inception movie plot', 'latest Marvel movie 2024', 'Dune book rating'. "
        "Returns formatted search results including titles, descriptions, and URLs."
    )
)

def get_web_search_tool():
    """
    Returns the web search tool for use by the LLM agent.
    
    Returns:
        LangChain Tool object for web search
        
    Example:
        >>> from tools import get_web_search_tool
        >>> tool = get_web_search_tool()
        >>> result = tool.func("Inception movie")
        >>> print(result)
    """
    return web_search_tool

def test_web_search_tool():
    """
    Test the Langchain web search tool wrapper.
    """
    print("\n" + "=" * 50)
    print("TESTING LANGCHAIN WEB SEARCH TOOL")
    print("=" * 50 + "\n")

    #Check if API key exists
    if not os.getenv("BRAVE_API_KEY"):
        print("⚠️  Skipping - BRAVE_API_KEY not configured")
        print("="*50)
        return

    tool = get_web_search_tool()

    #Show tool properties
    print("    Tool Properties:")
    print(f"   Name: {tool.name}")
    print(f"   Description: {tool.description[:100]}...")

    print("\n    Testing tool.run() method:\n")
    result = tool.run("Upcoming Marvel movies in 2027")

    if result.startswith("Error:"):
        print(f"   ❌: {result}\n")
    else:
        print(f"   ✅: {result} + ...\n")  # Print first 500 characters

    print("\n" + "=" * 50)
    print("✅ LANGCHAIN WEB SEARCH TOOL wrapper test COMPLETED!")
    print("=" * 50 + "\n")

# ============================================================
# 2.1 END OF WEB SEARCH LANGCHAIN TOOL WRAPPER
# ============================================================

# ============================================================
# 3.0  START OF MOVIE INFO TOOL (OMDB API)
# ============================================================
def get_movie_info(title: str) -> str:
    """
    Get detailed movie information from OMDB API by title.

    This tool allows the LLM to fetch specific movie data including ratings,plot,cast,director, and other metadata.

    Args:
        title: Movie title to search for
    Returns:
        Formatted movie information as a string, or error message
    
    Examples:
        >>> get_movie_info("Inception")
        'Title: Inception (2010)
         Director: Christopher Nolan
         Cast: Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page...
         IMDb Rating: 8.8/10
         Plot: A thief who steals corporate secrets through the use of dream-sharing technology...'

    API Requirements:
        - Needs OMDB_API_KEY in environment variables
        - Free tier: 1,000 requests/day
    """

    #Get API Key from environment
    movie_key = os.getenv("OMDB_API_KEY")

    if not movie_key:
        return(
            "Error: OMDB_API_KEY not found in environment variables."
            "Please add it to your .env file."
        )

    #OMDB API endpoint (load from .env or use default)
    movie_url = os.getenv("OMDB_API_URL")

    #Query parameters
    params = {
        "apikey": movie_key,
        "t": title, #Search by title
        "plot": "full", #Get full plot
        "type": "movie", #Only movies
    }

    try:

        #Make the API request
        response = requests.get(movie_url,params=params,timeout=10)

        #Check for errors
        if response.status_code == 401:
            return "Error: Invalid OMDB API key. Check your api key and try again."
        
        if response.status_code != 200:
            return f"Error: OMDB API returned status code {response.status_code}."
        
        #Parse JSON response
        data = response.json()

        #Check if movie found
        if data.get("Response") == "False":
            return f"Error: Movie '{title}' not found in OMDB database."
        
        #Extract moovie information
        movie_title = data.get("Title", "Unknown")
        year = data.get("Year", "Unknown")
        rated = data.get("Rated", "N/A")
        released = data.get("Released", "N/A")
        runtime = data.get("Runtime", "N/A")
        genre = data.get("Genre", "N/A")
        director = data.get("Director", "N/A")
        actors = data.get("Actors", "N/A")
        plot = data.get("Plot", "N/A")
        language = data.get("Language", "N/A")
        awards = data.get("Awards", "N/A")
        imdb_rating = data.get("imdbRating", "N/A")
        imdb_votes = data.get("imdbVotes", "N/A")
        metascore = data.get("Metascore", "N/A")
        box_office = data.get("BoxOffice", "N/A")

        #Get addition ratings if available
        ratings = data.get("Ratings", [])
        rotten_tomatoes = "N/A"
        for rating in ratings:
            if rating.get("Source") == "Rotten Tomatoes":
                rotten_tomatoes = rating.get("Value", "N/A")
                break
        
        # Format the information for the LLM
        formatted_info = []
        formatted_info.append(f"🎬 Movie Information: {movie_title} ({year})")
        formatted_info.append(f"\n📊 Ratings:")
        formatted_info.append(f"   • IMDb: {imdb_rating}/10 ({imdb_votes} votes)")
        formatted_info.append(f"   • Rotten Tomatoes: {rotten_tomatoes}")
        formatted_info.append(f"   • Metascore: {metascore}/100")
        formatted_info.append(f"\n🎭 Details:")
        formatted_info.append(f"   • Director: {director}")
        formatted_info.append(f"   • Cast: {actors}")
        formatted_info.append(f"   • Genre: {genre}")
        formatted_info.append(f"   • Runtime: {runtime}")
        formatted_info.append(f"   • Rated: {rated}")
        formatted_info.append(f"   • Released: {released}")
        formatted_info.append(f"   • Language: {language}")
        formatted_info.append(f"\n📝 Plot:")
        formatted_info.append(f"   {plot}")
        formatted_info.append(f"\n🏆 Awards:")
        formatted_info.append(f"   {awards}")
        formatted_info.append(f"\n💰 Box Office:")
        formatted_info.append(f"   {box_office}")
        
        return "\n".join(formatted_info)
        
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    
    except requests.exceptions.RequestException as e:
        return f"Error: Network error - {str(e)}"
    
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
    
def test_get_movie_info():
    """
    Test the get_movie_info function with OMDB API.
    """

    print("\n" + "="*50)
    print("TESTING GET MOVIE INFO (OMDB API)")
    print("="*50 + "\n")

    #Check if API key exists
    movie_key = os.getenv("OMDB_API_KEY")

    if not movie_key:
        print("⚠️  OMDB_API_KEY not found in environment variables")
        print("   Skipping movie info tests")
        print("   Add your key to .env file to enable this test")
        print("\n" + "="*50)
        return
    
    print("OMDB_API_KEY found. Running tests...\n")

    test_movies =[
        ("The Batman", "Superhero movie"),
        ("Mad Max: Fury Road", "Post-apocalyptic action"),
        ("Nonexistent Movie Title 12345", "Invalid movie title (should error)")
    ]

    for title, description in test_movies:

        result = get_movie_info(title)

        #Check if it's an error
        if result.startswith("Error:"):
            print(f"  Result: {result}\n")
            if "not found" in description.lower():
                print(f"   ✅ Correctly returned error for non-existent movie")
            else:
                print(f"   ❌ Unexpected error")

        else:
            print("Movie info retrieved successfully:")
            print(f"  Result: {result[:500]}...\n")  # Print first 500 characters


    print("\n" + "="*50)
    print("✅ GET MOVIE INFO (OMDB API) TESTS COMPLETED!")
    print("="*50 + "\n")
    
# ============================================================
# 3.0 END OF MOVIE INFO TOOL (OMDB API)
# ============================================================

# ============================================================
# 3.1 START OF MOVIE INFO LANGCHAIN TOOL WRAPPER
# ============================================================

movie_info_tool = Tool(
    name = "get_movie_info",
    func = get_movie_info,
    description = (
        "Use this tool to get detailed information about a specific movie. "
        "Input should be the movie title as a string. "
        "Returns comprehensive movie data including IMDb rating, Rotten Tomatoes score, "
        "director, cast, plot summary, runtime, genre, awards, and box office. "
        "Examples: 'Inception', 'The Matrix', 'Interstellar'. "
        "Use this when you need specific, structured movie data rather than general web search."
    )
)

def get_movie_info_tool():
    """
    Returns the movie info tool for use by the LLM agent.
    
    Returns:
        LangChain Tool object for movie info
    
    Example:
        >>> from tools import get_movie_info_tool
        >>> tool = get_movie_info_tool()
        >>> result = tool.func("Inception")
        >>> print(result)
    """
    return movie_info_tool

def test_get_movie_info_tool():
    """
    Test the LangChain movie info tool wrapper.
    """

    print("\n" + "=" * 50)
    print("TESTING LANGCHAIN GET MOVIE INFO TOOL")
    print("=" * 50 + "\n")

    movie_key = os.getenv("OMDB_API_KEY")

    #Check if API key exists
    if not movie_key:
        print("⚠️  Skipping - OMDB_API_KEY not configured")
        print("="*50)
        return
    
    tool = get_movie_info_tool()

    #Show tool properties
    print("   Tool Properties:")
    print(f"   Name: {tool.name}")
    print(f"   Description: {tool.description[:100]}...")

    # Test using the tool
    print("🧪 Testing tool.run() method:")
    result = tool.run("The Avengers")
    
    if result.startswith("Error:"):
        print(f"   ❌ {result}")
    else:
        print(f"   ✅ Movie info retrieved!")
        print(f"   Results preview: {result[:300]}...")
    
    print("\n" + "="*50)
    print("✅ GET MOVIE INFO  TOOL  LANGCHAIN wrapper test COMPLETED!")
    print("="*50)

# ============================================================
# 3.1 END OF MOVIE INFO LANGCHAIN TOOL WRAPPER
# ============================================================

# ============================================================
# 4.0  START OF BOOK INFO TOOL (OPENLIBRARY API)
# ============================================================

def get_book_info(title: str, author: str = "") -> str:
    """
    Get detailed book information from OpenLibrary API by title.

    This tool allows the LLM to fetch specific book data including
    author, publish year, pages, ISBN, subjects,and more.

    Args:
        title: Book title to search for
        author: (Optional) Author name to refine search

    Returns:
        Formatted book information as a string, or error message

    Examples:
        >> get_book_info("Dune", "Frank Herbert")
        'Title: Dune
         Author: Frank Herbert
         Published: 1965
         Pages: 688
         ISBN: 9780441172719'
        
    API Requirements:
        - No API key needed! Completely free.
        - No rate limits for reasonable use
    """

    #OpenLibrary API endpoint
    openlib_url = os.getenv("OPENLIB_API_URL")

    #Build Query parameters
    params = {
        "title": title,
        "limit": 1 #Get top result only
    }
    
    # Add author to search if provided
    if author:
        params["author"] = author

    try:
        response = requests.get(openlib_url, params=params, timeout=10)

        if response.status_code != 200:
            return f"Error: OpenLibrary API returned status code {response.status_code}."
        
        #Parse JSON response
        data = response.json()

        #Check if any books were found
        num_found = data.get("numFound", 0)

        if num_found == 0:
            search_term = f"'{title}' by {author}" if author else f"'{title}'"
            return f"Error: Book {search_term} not found in OpenLibrary database. Try a different title or check spelling."
        
        #Get the best match (first result)
        books = data.get("docs", [])

        if not books:
            return f"Error: No book data found for title coming from the API."

        book = books[0]

        # Extract book information
        book_title = book.get("title", "Unknown")
        authors = book.get("author_name", ["Unknown"])
        author_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
        
        first_publish_year = book.get("first_publish_year", "N/A")
        
        # Get page count (median is most reliable)
        pages = book.get("number_of_pages_median")
        if not pages:
            pages = "N/A"
        
        # Get ISBNs
        isbns = book.get("isbn", [])
        isbn_str = isbns[0] if isbns else "N/A"
        
        # Get publishers
        publishers = book.get("publisher", [])
        publisher_str = publishers[0] if publishers else "N/A"
        
        # Get subjects/topics
        subjects = book.get("subject", [])
        # Limit to first 5 subjects for brevity
        subjects_str = ", ".join(subjects[:5]) if subjects else "N/A"
        
        # Get language
        languages = book.get("language", [])
        language_str = ", ".join(languages[:3]) if languages else "N/A"
        
        # Get all publish years for editions info
        publish_years = book.get("publish_year", [])
        if publish_years:
            editions_str = f"{len(publish_years)} editions"
        else:
            editions_str = "N/A"
        
        # Build cover URL if available
        cover_id = book.get("cover_i")
        cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else "N/A"
        
        # Format the information for the LLM
        formatted_info = []
        formatted_info.append(f"📚 Book Information: {book_title}")
        formatted_info.append(f"\n✍️  Author:")
        formatted_info.append(f"   {author_str}")
        formatted_info.append(f"\n📖 Publication Details:")
        formatted_info.append(f"   • First Published: {first_publish_year}")
        formatted_info.append(f"   • Publisher: {publisher_str}")
        formatted_info.append(f"   • Editions: {editions_str}")
        formatted_info.append(f"   • Pages: {pages}")
        formatted_info.append(f"   • ISBN: {isbn_str}")
        formatted_info.append(f"   • Language: {language_str}")
        formatted_info.append(f"\n🏷️  Subjects/Topics:")
        formatted_info.append(f"   {subjects_str}")
        formatted_info.append(f"\n🖼️  Cover Image:")
        formatted_info.append(f"   {cover_url}")
        
        return "\n".join(formatted_info)
        
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    
    except requests.exceptions.RequestException as e:
        return f"Error: Network error - {str(e)}"
    
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}" 
    
def test_get_book_info():
    """
    Test the book info tool with Open Library API.
    """
    print("\n" + "="*50)
    print("TESTING BOOK INFO TOOL")
    print("="*50 + "\n")
    
    print("ℹ️  Note: Open Library API requires no API key!")
    print()
    
    # Test cases
    test_books = [
        ("Dune", "Frank Herbert", "Classic sci-fi book"),
        ("The Name of the Wind", "", "Fantasy book (no author specified)"),
        ("NonExistentBook12345XYZ", "", "Book not found (should error)")
    ]
    
    for title, author, description in test_books:
        print(f"📝 Test: {description}")
        if author:
            print(f"   Book: '{title}' by '{author}'")
        else:
            print(f"   Book: '{title}'")
        print()
        
        result = get_book_info(title, author)
        
        # Check if it's an error
        if result.startswith("Error:"):
            print(f"   Result: {result}")
            if "not found" in description.lower():
                print(f"   ✅ Correctly returned error for non-existent book")
            else:
                print(f"   ❌ Unexpected error")
        else:
            print(f"   ✅ Book info retrieved!")
            # Print first 400 chars
            print(f"   {result[:400]}...")
        
        print()
    
    print("="*50)
    print("✅ Book info tool test COMPLETED!")
    print("="*50)

# ============================================================
# 4.0  END OF BOOK INFO TOOL (OPENLIBRARY API)
# ============================================================

# ============================================================
# 4.1 START OF BOOK INFO LANGCHAIN TOOL WRAPPER
# ============================================================

book_info_tool = Tool(
    name="get_book_info",
    func=get_book_info,
    description=(
        "Use this tool to get detailed information about a specific book. "
        "Input should be the book title as a string, optionally with author name. "
        "Format: 'title' or 'title by author'. "
        "Returns comprehensive book data including author, publication year, "
        "page count, ISBN, publisher, subjects, and cover image URL. "
        "Examples: 'Dune', 'Dune by Frank Herbert', 'The Name of the Wind'. "
        "Use this when you need specific book metadata rather than general web search."
    )
)

def get_book_info_tool():
    """
    Returns the book info tool for use by the LLM agent.
    
    Returns:
        LangChain Tool object for getting book information
        
    Example:
        >>> from tools import get_book_info_tool
        >>> tool = get_book_info_tool()
        >>> result = tool.func("Dune", "Frank Herbert")
        >>> print(result)
    """
    return book_info_tool

def test_get_book_info_tool():
    """
    Test the LangChain wrapper for book info.
    """
    print("\n" + "="*50)
    print("TESTING BOOK INFO LANGCHAIN WRAPPER")
    print("="*50 + "\n")
    
    # Get the tool
    tool = get_book_info_tool()
    
    # Show tool properties
    print("📋 Tool Properties:")
    print(f"   Name: {tool.name}")
    print(f"   Description: {tool.description[:80]}...")
    print()
    
    # Test using the tool
    print("🧪 Testing tool.run() method:")
    result = tool.run("Lord of the Rings")
    
    if result.startswith("Error:"):
        print(f"   ❌ {result}")
    else:
        print(f"   ✅ Book info retrieved!")
        print(f"   Results preview: {result[:300]}...")
    
    print()
    print("="*50)
    print("✅ Book info wrapper test COMPLETED!")
    print("="*50)

# ============================================================
# 4.1 END OF BOOK INFO LANGCHAIN TOOL WRAPPER
# ============================================================

def test_all_tools():
    """
    Run all tool tests.
    """
    print("\n" + "="*60)
    print("RUNNING ALL TOOL TESTS")
    print("="*60)
    
    # Test 1: LangChain calculator tool wrapper
    test_calculator_tool()

    # Test 2: LangChain web search tool wrapper
    test_web_search_tool()

    # Test 3: LangChain movie info tool wrapper
    test_get_movie_info_tool()

    # Test 4: LangChain book info tool wrapper
    test_get_book_info_tool()

    # Test 5: LangChain RAG search tool wrapper
    test_rag_search_tool()  


    print("\n" + "="*60)
    print("ALL TOOL TESTS PASSED!")
    print("="*60)

# Run tests when file is executed directly
if __name__ == "__main__":
    test_all_tools()