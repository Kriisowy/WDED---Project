from Algorithm import Algorithm

if __name__ == "__main__":
    filepath = "iris.csv"
    alg = Algorithm()
    result = alg.discretize_file(filepath)
    print("Wyniki dyskretyzacji:")
    for key, value in result.items():
        print(f"{key}: {value}")
