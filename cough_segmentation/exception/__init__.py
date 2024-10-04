import os,sys


def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    file_name = os.path.split(file_name)[-1]

    error_message = f"Error occured in the file [{file_name}] at line number [{exc_tb.tb_lineno}]. Error message: {str(error)}"

    return error_message


class CoughSegmentationException(Exception):

    def __init__(self,error_message: str, error_detail: sys):

        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail)

    def __str__(self) -> str:
        return self.error_message