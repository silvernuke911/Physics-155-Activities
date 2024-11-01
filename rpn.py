import numpy as np

def compute_rpn(ops_stack, current_index=0):
    # Stack to hold numbers for RPN evaluation
    stack = []
    
    # Loop over each character in the input string
    for char in ops_stack:
        if char.isdigit():  # If the character is a digit, push it onto the stack
            stack.append(int(char))
        else:
            # If it's an operator, pop the top two elements from the stack
            if len(stack) < 2:
                raise ValueError("RPN syntax error: insufficient operands for operator.")
            
            b = stack.pop()
            a = stack.pop()
            
            # Apply the operation and push the result back onto the stack
            if char == '+':
                stack.append(a + b)
            elif char == '-':
                stack.append(a - b)
            elif char == '*':
                stack.append(a * b)
            elif char == '/':
                if b == 0:
                    raise ValueError("Division by zero.")
                stack.append(a / b)
            else:
                raise ValueError(f"Invalid operator: {char}")
    
    # After the loop, there should be exactly one element in the stack
    if len(stack) != 1:
        raise ValueError("RPN syntax error: extra operands or operators.")
    
    return stack.pop()

# Test with your example expressions
expressions = ['1', '11+', '34+5-','34-','24-5+','34+56+*','231*+9-','234*+']
for i, string in enumerate(expressions):
    print(f'Expression {i+1} = {string}')
print()
for i, string in enumerate(expressions):
    try:
        print(f'Result     {i+1} = {compute_rpn(string)}')
    except ValueError as e:
        print(f'Error      {i+1} = {e}')


