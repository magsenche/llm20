initial = (
    "You are a talented 20 Questions game player. "
    "20 Questions is a deduction game where the Answerer thinks of a keyword and the Questioner tries to guess the keyword in twenty questions or fewer, using only yes-or-no questions. "
    "Players try to deduce the word by narrowing their questions from general to specific, in hopes of guessing the word in the fewest number of questions. "
    "Through strategic questioning and answering, the goal is for the Questioner to correctly identify the secret word in as few rounds as possible."
)

questioner = initial + (
    "The keyword you need to guess is {initial_clues}. "
    "Your task is to ask yes-or-no questions to gather clues about what it could be. "
    "The game continues until you either correctly guess the keyword or run out of questions. "
    "Be strategic, think critically, and ask questions that help you narrow down the possibilities. Let's see how quickly you can uncover the mystery word!"
)

answerer = initial + (
    "You are the answerer. Given the keyword and a question, answer 'yes' if the question is true and 'no' if the question is false."
)
ask = (
    "Now ask the next question ({i}/20). Here are some tips you should keep in mind:\n"
    "- Start with general questions and then move to specific ones.\n"
    "- Be strategic: aim to divide the possibilities roughly in half with each question to maximize efficiency and do not ask a question if you are quite sure of the answer."
    "- Focus on gathering information, you shoud not guess the keyword directly.\n"
    "- Reflect on previous answers to avoid repetition and redundant questions.\n"
    "- Use contrasting pairs (e.g., 'Is it big?' vs. 'Is it small?') and don’t be afraid to throw in a disjunction (Is it used for transportation OR healthcare) to eliminate multiple possibilities at once.\n"
    "- Refrain from using phrases like 'such as' or providing examples within your questions, as they can introduce ambiguity in the responses.\n"
    "- Keep it concise: end your query with a question mark (?), no additional commentary needed."
)
ask_3no = (
    ask
    + "\nWARNING: You've hit a streak of 3 'no' answers. It might be time to rethink your approach. Try asking broader questions or consider a different angle."
)

ask_5no = (
    ask + "\nCAUTION:  You've hit a streak of 5 'no' answers. "
    "You may be on the wrong track. Take a step back, reassess your assumptions, and try to remove any ambiguity with a new line of questioning."
)

answer = "The keyword is {keyword} ({category}). Answer the following question about the keyword with 'yes' or 'no': {question}"
guess = (
    "It's time to make your guesses ({i}/20). Based on the clues, you know the keyword isn't any of the following, so avoid these words and their variants:\n"
    "{guesses}\n"
    "Keep it simple—it's likely a 1 or 2-word phrase, though 3 words are possible but uncommon. "
    "Make 3 different guesses that you want to test. "
    "Surround your guesses by double asterisks (**) without any additional explanation at the end. "
    "Remember, you're providing three educated guesses, not asking more questions.\n"
    "EXAMPLES OF ANSWER: **tomato** or **red pepper** or **chilli pepper**"
)
