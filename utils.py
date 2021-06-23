import numpy as np

class StepInfoPrinter:
    """A utility class to allow for easily printing messages to the terminal
    indicating progress in some process.
    """

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.__cur_step = 1
        self.__p = int(np.floor(np.log10(max_steps))) + 1
    
    def print(self, message: str, step: bool = True, end: str = ''):
        """Prints some info along with which step number the program is at.

        Args:
            message (str): 
                The info message to print.

            step (bool):
                Whether to print the current step number.
                Defaults to true.
            
            end (str, optional): 
                The character to end the line with. If this is the last message 
                being printed, end this with the line feed ('\n') character. 
                Defaults to ''.
        """
        if step:
            print(f"\x1b[1K\r[{self.__cur_step:{self.__p}}/{self.max_steps}] {message}", end=end)
            self.__cur_step += 1
        else:
            print(f"\x1b[1K\r{message}", end=end)

    def print_done(self):
        print(f"\x1b[1K\r[{self.max_steps}/{self.max_steps}] Done.")
