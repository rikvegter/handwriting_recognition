def print_info(message: str, end: str = ''):
    """Helper function that prints a status message to the terminal. Overwrites
    the current line, so that all messages are printed on the same line.

    Args: 
        message (str): 
            The info message to print.
        
        end (str, optional): 
            The character to end the line with. If this is the last message 
            being printed, end this with the line feed ('\n') character. 
            Defaults to ''.
    """
    print(f"\x1b[1K\r{message}", end=end)