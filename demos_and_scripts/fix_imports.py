import os

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Replacements
    new_content = content.replace("from app.runtime", "from symvion")
    new_content = new_content.replace("import app.runtime", "import symvion")
    new_content = new_content.replace("app.utils", "symvion.utils")

    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Fixed imports in {filepath}")

for root, dirs, files in os.walk('src/symvion'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))

print("Done.")
