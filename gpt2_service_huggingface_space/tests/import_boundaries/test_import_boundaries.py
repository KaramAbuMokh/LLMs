# Tests of the package boundaries: the search package stays free of torch, pydantic and fastapi, and only the app factory reads service_settings
import re  # import statements
import subprocess  # fresh interpreter processes
import sys  # the running interpreter
from pathlib import Path  # source folder

SERVICE_FOLDER = Path(__file__).resolve().parents[2]  # gpt2_service_huggingface_space
PACKAGE_FOLDER = SERVICE_FOLDER / "gpt2_demo_service"  # the service code


def list_modules_loaded_after_import(module_name: str) -> set[str]:  # top-level packages loaded by importing one module in a fresh process
    program = "import sys, %s; print(' '.join(sorted({name.split('.')[0] for name in sys.modules})))" % module_name  # prints loaded package names
    completed = subprocess.run([sys.executable, "-c", program], cwd=SERVICE_FOLDER, capture_output=True, text=True, check=True)  # fresh interpreter
    return set(completed.stdout.split())  # package names


def test_evidence_retriever_loads_no_torch_pydantic_or_fastapi():  # search runs without the heavy stacks
    loaded = list_modules_loaded_after_import("gpt2_demo_service.index_search.evidence_retriever")  # import it alone
    assert not loaded & {"torch", "pydantic", "fastapi", "starlette", "tiktoken", "huggingface_hub"}  # none of them
    assert "numpy" in loaded  # but numpy is used


def test_language_models_and_answers_do_not_load_fastapi():  # the web stack is only for web_api
    loaded = list_modules_loaded_after_import("gpt2_demo_service.grounded_answers.grounded_answer_service")  # import the answer path
    assert not loaded & {"pydantic", "fastapi", "starlette"}  # no web stack


def test_only_the_app_factory_imports_service_settings():  # D20
    importers = set()  # files that import the module
    for source_path in PACKAGE_FOLDER.rglob("*.py"):  # every service module
        if re.search(r"^\s*(from|import)\s+gpt2_demo_service\.service_settings\b", source_path.read_text(encoding="utf-8"), re.MULTILINE):  # imports it
            importers.add(source_path.relative_to(PACKAGE_FOLDER).as_posix())  # record the file
    assert importers == {"web_api/app_factory.py"}  # only the factory


def test_init_files_are_bare_package_markers():  # they re-export nothing
    for init_path in PACKAGE_FOLDER.rglob("__init__.py"):  # every package marker
        assert init_path.read_text(encoding="utf-8").strip() == ""  # empty


def test_lower_packages_never_import_upper_ones():  # dependency direction of DESIGN.md 5.1
    forbidden = {  # package -> packages it must not import
        "language_models": ("index_search", "index_loading", "grounded_answers", "web_api", "service_settings", "monitoring"),
        "index_loading": ("index_search", "grounded_answers", "web_api", "language_models", "service_settings"),
        "index_search": ("grounded_answers", "web_api", "language_models", "service_settings"),
        "grounded_answers": ("web_api", "service_settings"),
        "monitoring": ("web_api", "service_settings"),
        "service_startup": ("web_api", "service_settings"),
    }
    for package, banned in forbidden.items():  # every rule
        for source_path in (PACKAGE_FOLDER / package).glob("*.py"):  # every module of the package
            imported = re.findall(r"^\s*(?:from|import)\s+gpt2_demo_service\.(\w+)", source_path.read_text(encoding="utf-8"), re.MULTILINE)  # packages it imports
            assert not set(imported) & set(banned), "%s imports %s" % (source_path.name, sorted(set(imported) & set(banned)))  # clear failure
