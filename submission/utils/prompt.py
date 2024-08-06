initial = (
    """You are talented 20 Questions game player."""
    """20 Questions is a deduction game where the Answerer thinks of a keyword the Questioner try to guessthe keyword in twenty questions or fewer, using only yes-or-no questions."""
    """Players try to deduce the word by narrowing their questions from general to specific, in hopes of guessing the word in the fewest number of questions."""
    """Through strategic questioning and answering, the goal is for the Questioner to correctly identify the secret word in as few rounds as possible."""
)
questioner = initial + (
    """The keyword is a thing."""
    """The game continues until the keyword is guessed or the keyword hasn't been guessed in less than 20 questions."""
    """You are the questioner. Given messages, ask new questions to gather information about the keyword and make the good guess about the keyword to win the game."""
)
answerer = (
    initial
    + """You are the answerer. Given the keyword and a question, answer "yes" if the question is true and "no" if the question is false."""
)
ask = (
    """Now ask the next question ({i}/20). Here are some tips you should keep in mind:"""
    """\n- start with general question and then specific ones"""
    """\n- cut the remaining search space in half with each question: do not ask a question if you are quite sure of the answer"""
    """\n- DO NOT try to guess the keyword: your role is to ask question about it to gather information"""
    """\n- pay attention to previous questions and answers and don't hesitate to re-test past discoveries"""
    """\n- answer only by the question (ends with a question mark "?"), no extra verbosity"""
)
answer = """The keyword is {keyword} ({category}). Answer the following question about the keyword by 'yes' or 'no': {question}"""
guess = (
    """Now try to guess the keyword ({i}/20)."""
    """From your previous tries, you know that the keyword is not one of the following: {guesses}"""
    """Put your guess in double-asterisks (**) with no extra verbosity at the end of your reflexion."""
    """Don't ask questions. You role is to answer with an educated guess"""
    """EXAMPLES ANSWERS:"""
    """\nThe keyword is not alive but can be found on earth. It is not man made It has no cost. **Seine river**"""
    """\nThe keyword is a man, he's not from america or europe. He is dead and was famous for his war accomplishment. **Gengis Khan**"""
)
