import os

def check_and_prompt_overwrite(filename):
    extension = os.path.splitext(filename)[1]
    def get_user_input(prompt, valid_responses, invalid_input_limit=3):
        attempts = 0
        while attempts < invalid_input_limit:
            response = input(prompt).lower().strip()
            if response in valid_responses:
                return response
            print("Invalid input. Valid inputs are [ Y , N , YES , NO ]")
            attempts += 1
        print("Exceeded maximum invalid input limit. Operation aborted.")
        return 'ABORT'

    def handle_file_exists(filename):
        while True:
            response = get_user_input(f"{filename} already exists, do you want to overwrite it? (Y/N): ", ['yes', 'y', 'no', 'n'], 5)
            if response in ['yes', 'y']:
                print                       ('\nx---------------------WARNING---------------------x')
                sure_response = get_user_input("Are you really sure you want to OVERWRITE it? (Y/N): ", ['yes', 'y', 'no', 'n'], 5)
                if sure_response in ['yes', 'y']:
                    print("Proceeding with overwrite...")
                    return True, filename
                elif sure_response in ['no', 'n']:
                    print('Operation aborted.')
                    return False, filename
                elif sure_response == 'ABORT':
                    return False, filename
            elif response in ['no', 'n']:
                return handle_rename(filename)
            elif response == 'ABORT':
                return False, filename

    def handle_rename(filename):
        while True:
            rename_response = get_user_input('Would you like to rename it? (Y/N): ', ['yes', 'y', 'no', 'n'],3)
            if rename_response in ['yes', 'y', '1']:
                return get_new_filename()
            elif rename_response in ['no', 'n', '0']:
                print('Operation aborted.')
                return False, filename
            elif rename_response == 'ABORT':
                return False, filename

    def get_new_filename():
        while True:
            new_filename = input('Input the new name of the file: ').strip()
            # If the user doesn't specify an extension, add the original extension
            if not new_filename.endswith(extension):
                new_filename += extension
            if new_filename == ('ABORT' + extension):
                print('Operation aborted.')
                return False, new_filename
            if not os.path.isfile(new_filename):
                print(f'Proceeding with creation of {new_filename}')
                return True, new_filename
            print(f'{new_filename} already exists. Please put another file name.')
    if os.path.isfile(filename):
        return handle_file_exists(filename)
    return True, filename

import os

# Mock os.path.isfile to simulate file existence
def mock_isfile(path, exists=True):
    """Mock os.path.isfile behavior."""
    def isfile_mock(filename):
        return exists if filename == path else False
    os.path.isfile = isfile_mock  # Overwrite the real os.path.isfile with this mock

# Mock input to simulate user responses
def mock_input(responses):
    """Mock input function to return predefined responses."""
    def input_mock(prompt):
        print(prompt)  # Optionally print the prompt for clarity
        return responses.pop(0)
    return input_mock

# Test function
def test_check_and_prompt_overwrite():
    tests = [
        # Test 1: File does not exist
        {
            'filename': 'new_file.txt',
            'file_exists': False,
            'inputs': [],
            'expected': (True, 'new_file.txt'),
            'description': 'File does not exist, proceed without prompt.'
        },
        # Test 2: File exists, overwrite confirmed
        {
            'filename': 'existing_file.txt',
            'file_exists': True,
            'inputs': ['y', 'y'],
            'expected': (True, 'existing_file.txt'),
            'description': 'File exists, user confirms overwrite.'
        },
        # Test 3: File exists, rename confirmed
        {
            'filename': 'existing_file.txt',
            'file_exists': True,
            'inputs': ['n', 'y', 'new_file.txt'],
            'expected': (True, 'new_file.txt'),
            'description': 'File exists, user denies overwrite and renames.'
        },
        # Test 4: File exists, rename denied
        {
            'filename': 'existing_file.txt',
            'file_exists': True,
            'inputs': ['n', 'n'],
            'expected': (False, 'existing_file.txt'),
            'description': 'File exists, user denies both overwrite and rename.'
        },
        # Test 5: Exceeded invalid input limit
        {
            'filename': 'existing_file.txt',
            'file_exists': True,
            'inputs': ['invalid', 'invalid', 'invalid', 'invalid', 'invalid'],  # 5 invalid inputs
            'expected': (False, 'existing_file.txt'),
            'description': 'Exceeded invalid input limit, operation aborted.'
        },
        # Test 6: Abort during renaming step
        {
            'filename': 'existing_file.txt',
            'file_exists': True,
            'inputs': ['n', 'y', 'ABORT'],
            'expected': (False, 'ABORT.txt'),
            'description': 'Abort during renaming step.'
        }
    ]

    # Run all tests
    for i, test in enumerate(tests):
        # Mock os.path.isfile based on the test case
        mock_isfile(test['filename'], test['file_exists'])
        # Mock input() for this test
        input_values = test['inputs'].copy()  # Copy to avoid modifying original
        global input
        input = mock_input(input_values)  # Mock input function

        # Run the function with the test filename
        result = check_and_prompt_overwrite(test['filename'])
        
        # Check if result matches expected output
        assert result == test['expected'], f"Test {i+1} failed: {test['description']}.\nExpected {test['expected']}, got {result}"
        print(f"Test {i+1} passed: {test['description']}")

# Run the tests
test_check_and_prompt_overwrite()
