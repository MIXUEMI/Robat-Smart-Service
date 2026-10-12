from src.llm.chat_model import get_chat_model


def main():
    print("智能客服系统启动成功。")
    print("输入 exit 退出。")
    print("-" * 50)

    llm = get_chat_model()

    while True:
        user_input = input("用户：").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("系统已退出。")
            break

        if not user_input:
            print("请输入有效问题。")
            continue

        try:
            response = llm.invoke(user_input)
            print(f"客服助手：{response.content}")
            print("-" * 50)

        except Exception as e:
            print(f"调用大模型失败：{e}")


if __name__ == "__main__":
    main()