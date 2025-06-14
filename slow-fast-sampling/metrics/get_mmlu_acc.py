import json
from statistics import mean

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_exact_match_values(data):
    exact_match_values = []
    
    def traverse_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                if 'exact_match,get_response' in value:
                    exact_match_values.append(value['exact_match,get_response'])
                traverse_dict(value)
    
    traverse_dict(data)
    return exact_match_values

def main():
    file_path = 'results.json'  # Replace with the actual path
    data = load_json(file_path)
    exact_match_values = extract_exact_match_values(data)
    
    if exact_match_values:
        average = mean(exact_match_values)
        print(f"Average of exact_match,get_response: {average:.4f}")
        print(f"Number of values: {len(exact_match_values)}")
    else:
        print("No exact_match,get_response values found.")

if __name__ == "__main__":
    main()