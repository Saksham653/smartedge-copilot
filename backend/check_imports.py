import os
import sys
import importlib.util
import traceback

backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


files_with_errors = {}

for filename in os.listdir(backend_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        try:
            # Try to compile to check for syntax errors
            filepath = os.path.join(backend_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            compile(source, filepath, 'exec')
            
            # Now try to import
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ Successfully checked {filename}")
        except Exception as e:
            print(f"❌ Error in {filename}:")
            traceback.print_exc()
            files_with_errors[filename] = str(e)

print(f"\nTotal files with errors: {len(files_with_errors)}")
for f, e in files_with_errors.items():
    print(f" - {f}: {e}")
