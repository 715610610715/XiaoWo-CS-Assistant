"""联通小沃智能客服桌面客户端入口。"""

from app.bootstrap import build_application


def main() -> None:
    application = build_application()
    application.run()


if __name__ == "__main__":
    main()
