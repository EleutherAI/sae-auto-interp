from pydantic import create_model

def create_response_model(n: int):
    fields = {f'example_{i}': (int, ...) for i in range(n)}
    
    ResponseModel = create_model('ResponseModel', **fields)
    
    return ResponseModel
