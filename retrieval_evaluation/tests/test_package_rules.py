# Package-wide rules as tests: files stay under 150 lines, only the command touches the service, full-word names, a comment on every code line
import ast  # reads every module without importing it
import io  # token stream over source text
import tokenize  # finds comments
from pathlib import Path

PACKAGE_FOLDER = Path(__file__).resolve().parents[1]  # retrieval_evaluation/
SOURCE_PATHS = sorted(path for path in PACKAGE_FOLDER.rglob("*.py") if "__pycache__" not in path.parts)  # every module and test
FORBIDDEN_PACKAGES = ("gpt2_demo_service", "retrieval_encoders", "retrieval_index_builder")  # other work packages' code
COMMAND_FILE_NAME = "run_evaluation_command.py"  # the only module that may import the service at run time
MAX_LINES = 150  # hard limit per file
ALLOWED_SHORT_NAMES = {"i", "j", "n", "x", "y"}  # tight loop letters and well-known math
FORBIDDEN_NAME_PARTS = {"rag", "bm25", "rrf", "kv", "mrr", "ndcg", "ms", "cfg", "mgr", "usr", "req", "res", "resp", "db", "msg", "dir", "idx", "tmp", "proc",
                        "cnt", "pos", "qty", "arg", "param", "err", "fn", "func", "val", "num", "dev"}  # abbreviations the naming rule forbids


def find_runtime_imports(tree: ast.AST) -> list[str]:  # module names imported outside `if TYPE_CHECKING:` blocks
    found = []  # names
    def visit(node, inside_type_checking):  # walks the tree, remembering whether a TYPE_CHECKING block encloses the node
        for child in ast.iter_child_nodes(node):  # every child
            guarded = inside_type_checking or (isinstance(child, ast.If) and "TYPE_CHECKING" in ast.dump(child.test))  # a guarded block
            if isinstance(child, ast.Import) and not inside_type_checking:  # import x
                found.extend(alias.name for alias in child.names)  # names
            elif isinstance(child, ast.ImportFrom) and not inside_type_checking:  # from x import y
                found.append(child.module or "")  # module
            visit(child, guarded)  # go deeper
    visit(tree, False)  # start outside any block
    return found  # runtime imports


def collect_identifiers(tree: ast.AST) -> set[str]:  # every name the code defines or uses (no attribute names of other libraries)
    names = set()  # identifiers
    for node in ast.walk(tree):  # every node
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):  # definitions
            names.add(node.name)  # function or class name
        elif isinstance(node, ast.arg):  # parameters
            names.add(node.arg)  # parameter name
        elif isinstance(node, ast.Name):  # variables
            names.add(node.id)  # variable name
        elif isinstance(node, ast.Dict):  # dictionary keys we control
            names.update(key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.isidentifier())  # string keys
    return names  # all identifiers


def test_every_file_is_at_most_150_lines():  # the hard limit
    too_long = [path.name for path in SOURCE_PATHS if len(path.read_text(encoding="utf-8").splitlines()) > MAX_LINES]  # offenders
    assert too_long == []  # none


def test_only_the_command_imports_the_service_at_run_time():  # isolation of work packages (DESIGN.md §14)
    offenders = {}  # file -> forbidden imports
    for path in SOURCE_PATHS:  # every module
        if path.name == COMMAND_FILE_NAME or path.name == Path(__file__).name:  # the command and this rule file are allowed
            continue  # skip
        runtime = [name for name in find_runtime_imports(ast.parse(path.read_text(encoding="utf-8"))) if name.split(".")[0] in FORBIDDEN_PACKAGES]  # forbidden
        if runtime:  # found some
            offenders[path.name] = runtime  # record
    assert offenders == {}  # none


def test_names_are_full_words_without_forbidden_abbreviations():  # naming rule
    offenders = {}  # file -> bad names
    for path in SOURCE_PATHS:  # every module
        bad = sorted(name for name in collect_identifiers(ast.parse(path.read_text(encoding="utf-8")))
                     if (len(name.strip("_")) == 1 and name.strip("_") not in ALLOWED_SHORT_NAMES) or (name != "tmp_path" and set(name.lower().split("_")) & FORBIDDEN_NAME_PARTS))  # bad
        if bad:  # found some
            offenders[path.name] = bad  # record
    assert offenders == {}  # none


def find_uncommented_statements(source: str) -> list[int]:  # first lines of statements whose header carries no comment (a docstring counts as one)
    comment_lines = {token.start[0] for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT}  # lines with a comment
    bare = []  # offending line numbers
    for node in ast.walk(ast.parse(source)):  # every node
        if not isinstance(node, ast.stmt):  # only statements need a comment
            continue  # skip expressions and clauses
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):  # a docstring explains itself
            continue  # skip
        body = getattr(node, "body", None)  # compound statements have a body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node) is not None:  # a docstring states the job
            continue  # skip: the docstring is the comment of the definition
        header_end = body[0].lineno - 1 if isinstance(body, list) and body else node.end_lineno  # a compound statement's header ends where its body starts
        if not any(line in comment_lines for line in range(node.lineno, header_end + 1)) and (node.lineno - 1) not in comment_lines:  # no comment on or above it
            bare.append(node.lineno)  # report
    return sorted(bare)  # in file order


def test_every_statement_has_a_comment():  # the house rule: a short comment on each line of code (continuation lines share their statement's comment)
    missing = {path.name: find_uncommented_statements(path.read_text(encoding="utf-8"))[:8] for path in SOURCE_PATHS}  # per file
    assert {name: lines for name, lines in missing.items() if lines} == {}  # none
