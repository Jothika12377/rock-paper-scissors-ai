import random

CHOICES = ["rock", "paper", "scissors"]

def get_winner(user, comp):
    if user == comp:
        return "tie"
    if (user == "rock" and comp == "scissors") or \
       (user == "paper" and comp == "rock") or \
       (user == "scissors" and comp == "paper"):
        return "user"
    return "comp"

def main():
    user_score = 0
    comp_score = 0

    print("🎮 Rock Paper Scissors Game")
    print("Type: rock / paper / scissors")
    print("Type: quit to stop\n")

    while True:
        user = input("Your choice: ").strip().lower()

        if user == "quit":
            break

        # allow shortcuts
        if user in ["r", "p", "s"]:
            user = {"r": "rock", "p": "paper", "s": "scissors"}[user]

        if user not in CHOICES:
            print("❌ Invalid! Please type rock, paper, or scissors.\n")
            continue

        comp = random.choice(CHOICES)
        print(f"Computer chose: {comp}")

        result = get_winner(user, comp)

        if result == "tie":
            print("🤝 It's a tie!")
        elif result == "user":
            user_score += 1
            print("✅ You win!")
        else:
            comp_score += 1
            print("😅 Computer wins!")

        print(f"Score -> You: {user_score} | Computer: {comp_score}\n")

    print("\nGame Over!")
    print(f"Final Score -> You: {user_score} | Computer: {comp_score}")

if __name__ == "__main__":
    main()
