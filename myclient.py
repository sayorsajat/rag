# test_client_fastapi.py
import requests
import json

# Adres URL naszego nowego endpointu w FastAPI
SERVER_URL = "http://127.0.0.1:8000/ask"

print("🍹 Klient Koktajlowego Bota")
print("Napisz 'exit', aby zakończyć.")
print("-" * 40)


def ask_server(prompt: str) -> str:
    payload = {
        "text": prompt
    }

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=120)

        response.raise_for_status()

        data = response.json()

        return data.get("response", "Błąd: Nie znaleziono klucza 'response' w odpowiedzi.")

    except requests.exceptions.HTTPError as e:
        return f"Błąd HTTP: {e.response.status_code} - {e.response.text}"
    except requests.exceptions.RequestException as e:
        return f"Błąd połączenia z serwerem: {e}"
    except json.JSONDecodeError:
        return "Błąd: Nie udało się odczytać odpowiedzi serwera (niepoprawny JSON)."


while True:
    user_input = input("Ty: ")
    if user_input.lower() == "exit":
        break

    agent_response = ask_server(user_input)
    print(f"Bot: {agent_response}")

print("Do zobaczenia!")