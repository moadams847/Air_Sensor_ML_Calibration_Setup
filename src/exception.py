import sys 
from src.logger import logging

def error_message_details(error, error_details:sys):
    _,_,exe_tb = error_details.exc_info()
    file_name=exe_tb.tb_frame.f_code.co_filename
    line_number = exe_tb.tb_lineno 

    error_message = f"Error occured in python script {file_name} line number {line_number} error message {str(error)}"
    return error_message

    
class CustomeException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details=error_details)

    def __str__(self):
        return self.error_message

