# 🔍 Opis działania algorytmu
Algorytm implementuje zachłanną dyskretyzację wstępującą (bottom_up_discretization) na zbiorze danych. Celem jest pogrupowanie przykładów (próbek) w przedziały, tak aby zmaksymalizować liczbę par próbek należących do różnych klas, które są w różnych przedziałach — innymi słowy: maksymalizować separację klas między przedziałami.

# 🧩 Szczegóły poszczególnych części:
## Wczytanie danych:

load_dataset() ładuje dane z pliku CSV i usuwa wiersze z brakującymi wartościami.

### Podział danych:

separate_features_and_labels() dzieli dane na cechy (features) i etykiety klas (labels).

### Kodowanie danych kategorycznych:

encode_features() i encode_labels() kodują wartości tekstowe na liczby całkowite, aby model mógł je przetwarzać.

### Standaryzacja cech:

standardize_features() przekształca dane tak, by miały średnią 0 i odchylenie standardowe 1.

### Tworzenie początkowych przedziałów:

build_initial_intervals() tworzy jeden przedział na każdą próbkę.

### Obliczanie liczby par z różnych klas w różnych przedziałach:

count_separated_pairs() liczy, ile jest takich par — im więcej, tym lepsza separacja.

### Łączenie przedziałów:

merge_intervals() scala dwa sąsiednie przedziały.

### Algorytm główny bottom_up_discretization():

Iteracyjnie łączy przedziały, wybierając za każdym razem takie połączenie, które nie pogarsza liczby separowanych par.

Przestaje działać, gdy nie da się już poprawić wyniku.