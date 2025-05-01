from typing import Tuple, List
import csv

from neural import NeuralNet


def parse_line(tokens: list) -> Tuple[List[float], List[float]]:
    """Convert a row of the Open University assessment data into inputs and output"""
    # Assuming first column is 'final_result' with values 'Pass','Fail','Distinction'
    outcome = tokens[0]
    # Map final result to numeric output
    output = [1.0 if outcome == 'Pass' else 0.5 if outcome == 'Distinction' else 0.0]
    # Convert the rest of columns to float (skipping non-numeric if present)
    features = []
    for val in tokens[1:]:
        try:
            features.append(float(val))
        except ValueError:
            # skip or encode categorical here
            features.append(0.0)
    return features, output


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Scale input features to [0,1] range across data set"""
    if not data:
        return data
    n_features = len(data[0][0])
    mins = [float('inf')] * n_features
    maxs = [float('-inf')] * n_features
    for x, _ in data:
        for i, v in enumerate(x):
            if v < mins[i]: mins[i] = v
            if v > maxs[i]: maxs[i] = v
    for x, _ in data:
        for i in range(n_features):
            if maxs[i] > mins[i]:
                x[i] = (x[i] - mins[i]) / (maxs[i] - mins[i])
            else:
                x[i] = 0.0
    return data


def load_data(path: str):
    """Reads the CSV and returns list of (inputs, output) tuples"""
    data = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # skip header
        for row in reader:
            if row:
                inp, out = parse_line(row)
                data.append((inp, out))
    return data


def main():
    # Path to the downloaded CSV file
    csv_path = 'assessments.csv'

    # Load and preprocess data
    data = load_data(csv_path)
    data = normalize(data)

    # Define network: inputs = num features, hidden = e.g. 10, outputs = 1
    num_inputs = len(data[0][0])
    nn = NeuralNet(num_inputs, 10, 1)

    # Train
    nn.train(data, learning_rate=0.1, momentum_factor=0.1, iters=50000, print_interval=5000)

    # Test on training data
    for inp, expected, actual in nn.test_with_expected(data):
        print(f"in: {inp[:5]}... desired: {expected}, actual: {actual}")

if __name__ == '__main__':
    main()
