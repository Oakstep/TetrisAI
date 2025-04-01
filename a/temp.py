
import requests
import numpy as np

def get_player_stats(player_name):
    """Fetch player stats from the Skyblock API."""
    api_url = f"https://sky.shiiyu.moe/api/v2/dungeons/{player_name}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_class_average_difference(player_name):
    """Calculate the difference between the player's class average and 50."""
    player_stats = get_player_stats(player_name)
    if player_stats and 'dungeons' in player_stats:
        dungeons_data = player_stats['dungeons']
        class_average = dungeons_data.get('average_class_level', 0)
        difference = 50 - class_average
        return difference
    else:
        print("Could not retrieve player stats or dungeons data.")
        return None

def main():
    player_name = input("Enter the player's name: ")
    difference = calculate_class_average_difference(player_name)
    if difference is not None:
        print(f"The player is {abs(difference)} levels {'below' if difference > 0 else 'above'} the class average of 50.")

def search_for_xp(json_data, target_classes):
    """Search for 'xp' within the 'level' key inside the 'classes' structure."""
    results = {}

    def recursive_search(data):
        if isinstance(data, dict):  # If the current element is a dictionary
            for key, value in data.items():
                if key == "classes" and isinstance(value, dict):  # Found the 'classes' structure
                    for class_name, class_data in value.items():
                        if class_name.lower() in (name.lower() for name in target_classes):  # Match class name case-insensitively
                            if class_name not in results:
                                results[class_name] = []
                            # Look for 'level' key and extract 'xp' if available
                            if "level" in class_data and isinstance(class_data["level"], dict):
                                level_data = class_data["level"]
                                if "xp" in level_data:
                                    results[class_name].append({"xp": level_data["xp"]})
                # Recurse further into nested dictionaries
                recursive_search(value)
        elif isinstance(data, list):  # If the current element is a list
            for item in data:
                recursive_search(item)

    # Start the recursive search
    recursive_search(json_data)
    return results

def find_max_xp_for_classes(results):
    """Find the maximum 'xp' value for each class in the results dictionary."""
    max_results = {}
    for class_name, xp_list in results.items():
        max_xp = max(entry['xp'] for entry in xp_list if 'xp' in entry)  # Extract and find the maximum 'xp' value
        max_results[class_name] = max_xp
    return max_results

def calc_cata_50(results):
    max_xp = 569809640  # Maximum XP required
    xp_selected_class = 362000  # XP gained per run with selected class
    xp_different_class = 89000  # XP gained per run with different class
    A = np.array([
        [xp_selected_class, xp_different_class, xp_different_class, xp_different_class, xp_different_class],
        [xp_different_class, xp_selected_class, xp_different_class, xp_different_class, xp_different_class],
        [xp_different_class, xp_different_class, xp_selected_class, xp_different_class, xp_different_class],
        [xp_different_class, xp_different_class, xp_different_class, xp_selected_class, xp_different_class],
        [xp_different_class, xp_different_class, xp_different_class, xp_different_class, xp_selected_class]
    ])
    b = np.zeros(5)
    idx = 0
    for key in results:
        b[idx] = max_xp-results[key]
        idx+=1
    solved = np.linalg.solve(A, b)
    idx = 0
    for key in results:
        results[key] = solved[idx]
        idx+=1
    return results
def calc_total_runs(results):
    return np.round(sum(results.values()))

target_classes = ["mage", "tank", "healer", "berserk", "archer"]
print(calc_cata_50(find_max_xp_for_classes(search_for_xp(get_player_stats("Enshi"), target_classes=target_classes))))
print(calc_total_runs(calc_cata_50(find_max_xp_for_classes(search_for_xp(get_player_stats("Enshi"), target_classes=target_classes)))))