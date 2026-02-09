# House Price Prediction using Machine Learning
# Predicting house prices based on features
# Created by: [Your Name]
# Algorithm: Linear Regression (from scratch)

# Sample house data
def get_house_data():
    # Training data: [Area, Bedrooms, Bathrooms, Age, Location, Price]
    training_data = [
        [1200, 2, 1, 5, 7, 250000],
        [1500, 3, 2, 3, 8, 320000],
        [1800, 3, 2, 10, 6, 280000],
        [2000, 4, 3, 2, 9, 450000],
        [1000, 2, 1, 15, 5, 180000],
        [2200, 4, 3, 1, 9, 520000],
        [1400, 3, 2, 8, 7, 290000],
        [1600, 3, 2, 5, 8, 340000],
        [1900, 4, 2, 12, 6, 310000],
        [2500, 5, 4, 3, 10, 600000],
        [1100, 2, 1, 20, 4, 160000],
        [1700, 3, 2, 6, 7, 310000],
        [2100, 4, 3, 4, 8, 480000],
        [1300, 2, 2, 7, 6, 240000],
        [1950, 4, 3, 9, 7, 390000],
        [1450, 3, 2, 11, 6, 270000],
        [2300, 5, 3, 2, 9, 550000],
        [1250, 2, 1, 14, 5, 210000],
        [1850, 3, 2, 5, 8, 360000],
        [2050, 4, 3, 3, 9, 470000]
    ]
    return training_data

# Simple Linear Regression Model
class HousePricePredictor:
    def __init__(self):
        self.weights = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.bias = 0.0
        self.learning_rate = 0.00000001  # Very small for numerical stability
        self.trained = False
    
    def predict(self, features):
        prediction = self.bias
        for i in range(len(features)):
            prediction = prediction + (self.weights[i] * features[i])
        return prediction
    
    def train(self, data, epochs=1000):
        print("Training the model...")
        print("-" * 60)
        
        num_samples = len(data)
        
        for epoch in range(epochs):
            total_error = 0.0
            
            for house in data:
                features = house[:-1]
                actual_price = house[-1]
                
                predicted_price = self.predict(features)
                error = actual_price - predicted_price
                total_error = total_error + (error * error)
                
                # Update weights
                for i in range(len(self.weights)):
                    self.weights[i] = self.weights[i] + (self.learning_rate * error * features[i])
                
                self.bias = self.bias + (self.learning_rate * error)
            
            avg_error = (total_error / num_samples) ** 0.5
            
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Error: ${avg_error:,.0f}")
        
        self.trained = True
        print("-" * 60)
        print("Training complete!")
        print()
    
    def show_weights(self):
        feature_names = ['Area', 'Bedrooms', 'Bathrooms', 'Age', 'Location']
        print("\nMODEL WEIGHTS:")
        print("-" * 60)
        for i in range(len(self.weights)):
            print(f"{feature_names[i]:15s} : {self.weights[i]:15.4f}")
        print(f"{'Bias':15s} : {self.bias:15.4f}")
        print()

# Calculate model accuracy
def calculate_accuracy(model, data):
    total_error = 0.0
    
    for house in data:
        features = house[:-1]
        actual_price = house[-1]
        predicted_price = model.predict(features)
        error = abs(actual_price - predicted_price)
        total_error = total_error + error
    
    avg_error = total_error / len(data)
    return avg_error

# Test the model
def test_model(model, data):
    print("\nTESTING MODEL - Sample Predictions:")
    print("-" * 70)
    print(f"{'Actual Price':>15s} | {'Predicted':>15s} | {'Difference':>15s}")
    print("-" * 70)
    
    for i in range(min(5, len(data))):
        house = data[i]
        features = house[:-1]
        actual_price = house[-1]
        predicted_price = model.predict(features)
        difference = abs(actual_price - predicted_price)
        
        print(f"${actual_price:>14,.0f} | ${predicted_price:>14,.0f} | ${difference:>14,.0f}")
    
    print("-" * 70)
    print()

# Predict new house price
def predict_new_house(model):
    print("\n" + "=" * 60)
    print("              PREDICT NEW HOUSE PRICE")
    print("=" * 60)
    print()
    
    if not model.trained:
        print("Error: Model not trained!")
        return
    
    print("Enter house details:")
    
    try:
        area = int(input("  Area (sq ft): "))
        bedrooms = int(input("  Bedrooms: "))
        bathrooms = int(input("  Bathrooms: "))
        age = int(input("  Age (years): "))
        location = int(input("  Location (1-10): "))
        
        features = [area, bedrooms, bathrooms, age, location]
        predicted_price = model.predict(features)
        
        print()
        print("-" * 60)
        print("PREDICTION RESULT:")
        print("-" * 60)
        print(f"Area         : {area} sq ft")
        print(f"Bedrooms     : {bedrooms}")
        print(f"Bathrooms    : {bathrooms}")
        print(f"Age          : {age} years")
        print(f"Location     : {location}/10")
        print()
        print(f"Predicted Price: ${predicted_price:,.2f}")
        print("-" * 60)
        
    except ValueError:
        print("Error: Please enter valid numbers!")
    except Exception as e:
        print(f"Error: {e}")

# Compare sample houses
def compare_houses(model):
    print("\n" + "=" * 60)
    print("              SAMPLE HOUSE COMPARISONS")
    print("=" * 60)
    print()
    
    houses = [
        ['Small House', [1000, 2, 1, 10, 5]],
        ['Medium House', [1800, 3, 2, 5, 7]],
        ['Large House', [2500, 5, 4, 2, 10]]
    ]
    
    for house in houses:
        name = house[0]
        features = house[1]
        price = model.predict(features)
        print(f"{name:15s} : ${price:,.0f}")
    
    print()

# Feature importance
def show_feature_importance(model):
    print("\n" + "=" * 60)
    print("              FEATURE IMPORTANCE")
    print("=" * 60)
    print()
    
    features = ['Area', 'Bedrooms', 'Bathrooms', 'Age', 'Location']
    
    total = 0.0
    for w in model.weights:
        total = total + abs(w)
    
    if total == 0:
        print("Model not trained yet!")
        return
    
    print("Impact of each feature on price:")
    print("-" * 60)
    
    for i in range(len(model.weights)):
        importance = (abs(model.weights[i]) / total) * 100
        bar = "█" * int(importance / 2)
        print(f"{features[i]:12s} : {bar} {importance:.1f}%")
    
    print()

# Save report
def save_report(model, data, filename):
    try:
        f = open(filename, 'w')
        
        f.write("HOUSE PRICE PREDICTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MODEL WEIGHTS:\n")
        f.write("-" * 60 + "\n")
        features = ['Area', 'Bedrooms', 'Bathrooms', 'Age', 'Location']
        for i in range(len(model.weights)):
            f.write(f"{features[i]}: {model.weights[i]:.4f}\n")
        f.write(f"Bias: {model.bias:.4f}\n\n")
        
        f.write("SAMPLE PREDICTIONS:\n")
        f.write("-" * 60 + "\n")
        
        for i in range(min(10, len(data))):
            house = data[i]
            feat = house[:-1]
            actual = house[-1]
            pred = model.predict(feat)
            
            f.write(f"\nHouse {i+1}:\n")
            f.write(f"  Actual: ${actual:,.0f}\n")
            f.write(f"  Predicted: ${pred:,.0f}\n")
            f.write(f"  Error: ${abs(actual-pred):,.0f}\n")
        
        avg_err = calculate_accuracy(model, data)
        f.write(f"\nAverage Error: ${avg_err:,.0f}\n")
        
        f.close()
        print(f"\nReport saved as '{filename}'")
        
    except Exception as e:
        print(f"Error saving file: {e}")

# Main program
def main():
    print("=" * 60)
    print("         HOUSE PRICE PREDICTION SYSTEM")
    print("         Machine Learning - Linear Regression")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading training data...")
    house_data = get_house_data()
    print(f"Loaded {len(house_data)} houses")
    print()
    
    # Create and train model
    model = HousePricePredictor()
    model.train(house_data, epochs=1000)
    
    # Show weights
    model.show_weights()
    
    # Test model
    test_model(model, house_data)
    
    # Show accuracy
    avg_error = calculate_accuracy(model, house_data)
    print(f"Model Average Error: ${avg_error:,.0f}")
    print()
    
    # Feature importance
    show_feature_importance(model)
    
    # Compare houses
    compare_houses(model)
    
    # Menu loop
    while True:
        print()
        print("=" * 60)
        print("MENU:")
        print("1. Predict price for new house")
        print("2. Show model weights")
        print("3. Test model accuracy")
        print("4. Show feature importance")
        print("5. Save report to file")
        print("6. Exit")
        print("=" * 60)
        
        choice = input("\nChoose option (1-6): ")
        
        if choice == '1':
            predict_new_house(model)
        elif choice == '2':
            model.show_weights()
        elif choice == '3':
            test_model(model, house_data)
        elif choice == '4':
            show_feature_importance(model)
        elif choice == '5':
            fname = input("Enter filename: ")
            if fname == "":
                fname = "report"
            fname = fname + ".txt"
            save_report(model, house_data, fname)
        elif choice == '6':
            print()
            print("=" * 60)
            print("Thank you for using House Price Prediction!")
            print("=" * 60)
            break
        else:
            print("Invalid choice!")

# Run the program
if __name__ == "__main__":
    main()
