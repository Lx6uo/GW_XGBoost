import logging


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Hello from code!")


if __name__ == "__main__":
    main()
