try:
    from .evaluator import main
except ImportError:
    from evaluator import main


if __name__ == "__main__":
    main()
