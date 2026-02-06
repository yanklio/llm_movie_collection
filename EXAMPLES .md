# System Usage Examples (Proof of Execution)

The logs below represent actual session data from the program, demonstrating key functionalities: error handling, RAG, sentiment analysis, and database management.

## 1. Personalization & Sentiment Analysis (Human-in-the-Loop)
**Objective:** Demonstrates that the system understands user reviews and can correct typos ("matix" -> "The Matrix").

**Input:**
```bash
python main.py -v "Add review about matix - No so good film, actually not so good"
```

**Output (Log):**
```text
Query: Add review about matix - No so good film, actually not so good
⚡️ Dispatcher → Deciding which agent to call (qwen/qwen3-32b)
→ librarian
Processing: Add review about matrix - No so good film, actually not so good
Tool: check_entity({'title': 'Matrix'})
...
It looks like "Matrix" isn't in your collection, but you have "The Matrix" listed. Would you like me to 
add "The Matrix" to your collection now? This would help connect your review to the correct entry.
```

---

## 2. RAG (Retrieval Augmented Generation) with Context
**Objective:** Shows that the `MovieCritic` agent considers the user's negative opinion when generating recommendations.

**Input:**
```bash
python main.py -v "Recommend movie based on my reviews"
```

**Output (Log):**
```text
Processing query: Recommend a movie based on my reviews
✓ Found 1 relevant reviews for context
✓ Found 5 relevant movies
...
Based on your review of 'The Matrix' (which you didn't enjoy), I recommend trying a different genre. 
Consider watching 'Inception' for a mind-bending sci-fi experience or 'The Grand Budapest Hotel' for a 
whimsical comedy.
```

---

## 3. Complex Scenario (Chain of Thought / Workflow)
**Objective:** Demonstrates the full data lifecycle (Add -> Verify -> Use -> Delete).

**Input (Sequence):**
1. `Add The Dark Knight to my collection`
2. `Do I have The Dark Knight?`
3. `Recommend a superhero movie from my collection`
4. `Remove The Dark Knight`

**Output (Truncated Log):**
```text
--- STEP 1: ADDING ---
Query: Add The Dark Knight to my collection
→ scout (Fetching data from TMDB)
→ librarian (Saving to Vector Store)
✓ Stored: The Dark Knight

--- STEP 2: VERIFICATION ---
Query: Do I have The Dark Knight?
Result: Yes, you have The Dark Knight (2008) in your collection.

--- STEP 3: RAG (RECOMMENDATION) ---
Query: Recommend a superhero movie from my collection
Result:
- The Dark Knight – A classic superhero film featuring Batman fighting crime in Gotham.
- Guardians of the Galaxy – A Marvel superhero adventure set in space.

--- STEP 4: CLEANUP ---
Query: Remove The Dark Knight
Tool: delete_entity({'title': 'The Dark Knight'})
Result: ✅ “The Dark Knight” has been removed from your collection.
```

---

## 4. Error Handling & Safety
**Objective:** Shows that the system does not "hallucinate" when functionality is missing.

**Input:**
```bash
python main.py -v "What reviews i have"
```

**Output (Log):**
```text
Query: What reviews i have
⚡️ Dispatcher → Deciding which agent to call
→ librarian
Processing: get all reviews
...
The system cannot retrieve all reviews as this functionality is not supported. You can add new reviews 
using the librarian agent or ask for recommendations using the movie_critic agent.
```

---

## 5. Collection Overview (Watchlist)
**Input:**
```bash
python main.py "What movies do i have in watchlist"
```

**Output:**
```text
Your watchlist contains 36 movies across various genres including Action/Adventure, Sci-Fi, Horror, and 
more. Some highlighted titles include Inception (2010), Kill Bill: Vol. 1 (2003), Jurassic Park (1993), 
and Interstellar (2014).
```

