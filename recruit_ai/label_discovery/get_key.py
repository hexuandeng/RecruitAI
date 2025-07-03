def get_deepseek_key():
    """Read the key from the specified file"""
    try:
        with open("deepseek_key", 'r', encoding='utf-8') as file:
            secret_key = file.read().strip()
        return secret_key
    except FileNotFoundError:
        print("Key file not found!")
        return None
    except Exception as e:
        print(f"An error occurred while reading the key: {e}")
        return None


def get_gpt_key():
    """Read the key from the specified file"""
    try:
        with open("gpt_key", 'r', encoding='utf-8') as file:
            secret_key = file.read().strip() 
        return secret_key
    except FileNotFoundError:
        print("Key file not found!")
        return None
    except Exception as e:
        print(f"An error occurred while reading the key: {e}")
        return None


if __name__ == "__main__":
    path = "deepseek_key"  # The file path where the key is stored
    key = get_deepseek_key()
    if key:
        print("Successfully obtained the key:", key)
