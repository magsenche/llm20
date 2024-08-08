initial = (
    "You are a talented 20 Questions game player. "
    "20 Questions is a deduction game where the Answerer thinks of a keyword and the Questioner tries to guess the keyword in twenty questions or fewer, using only yes-or-no questions. "
    "Players try to deduce the word by narrowing their questions from general to specific, in hopes of guessing the word in the fewest number of questions. "
    "Through strategic questioning and answering, the goal is for the Questioner to correctly identify the secret word in as few rounds as possible."
)

questioner = initial + (
    "The keyword is a thing. "
    "The game continues until the keyword is guessed or the keyword hasn’t been guessed in fewer than 20 questions. "
    "You are the questioner. Given messages, ask new questions to gather information about the keyword and make a good guess about the keyword to win the game."
)
answerer = initial + (
    "You are the answerer. Given the keyword and a question, answer 'yes' if the question is true and 'no' if the question is false."
)
ask = (
    "Now ask the next question ({i}/20). Here are some tips you should keep in mind:\n"
    "- Start with general questions and then move to specific ones.\n"
    "- Be strategic: aim to divide the possibilities roughly in half with each question to maximize efficiency and do not ask a question if you are quite sure of the answer."
    "- DO NOT try to guess the keyword: your role is to ask questions about it to gather information.\n"
    "- Consider the context and previous answers to avoid redundant question.\n"
    "- Use contrasting pairs (e.g., 'Is it big?' vs. 'Is it small?') and don’t be afraid to throw in a disjunction (Is it used for transportation OR healthcare) to eliminate multiple possibilities at once.\n"
    "- Answer only with a question (ending with a question mark '?'), no extra verbosity."
)
ask_3no = (
    ask
    + "\nWARNING: You received 3 'no' answers in a row. Consider refocusing your strategy by asking broader questions or switching to a different line of inquiry"
)
answer = "The keyword is {keyword} ({category}). Answer the following question about the keyword with 'yes' or 'no': {question}"
guess = (
    "Now try to guess the keyword ({i}/20). You know it is not one of the following (don't try them again):\n"
    "{guesses}\n"
    "Keep it simple, it is very likely to be 1 or 2 words, 3 words is possible but unlikely. "
    "Surround your guess by double asterisks (**) with no extra verbosity at the end of your reflection. "
    "Don’t ask questions. Your role is to answer with an educated guess.\n"
    "EXAMPLES OF ANSWER: 'The keyword is not man made, is a type of food and can grow in Europe. It's red. **Tomato**'"
)
