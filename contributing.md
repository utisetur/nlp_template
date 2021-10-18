### Contributing

If you want to submit a bug or have a feature request create an issue at https://github.com/sberbank-ai-lab/pl_template/issues

Contributing is done using pull requests (direct commits into master branch are disabled).

## To create a pull request:
1. Clone it.
2. Install pre-commit hook, initialize it and install requirements:

```shell
pip install pre-commit
pip install -r requirements.txt
pre-commit install
```

3. Make  changes to the code.
4. Run tests:

```shell
pytest
```

5. Push code to your forked repo and create a pull request
