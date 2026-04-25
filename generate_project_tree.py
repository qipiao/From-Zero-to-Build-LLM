import os

def get_project_tree(root='.', prefix='', ignore_list=['.git', '__pycache__', '.idea', '.venv']):
    contents = sorted(os.listdir(root))
    contents = [c for c in contents if c not in ignore_list]

    dirs = []
    files = []
    for node in contents:
        path = os.path.join(root, node)
        if os.path.isdir(path):
            dirs.append(node)
        else:
            files.append(node)

    contents = dirs + files
    tree_lines = []

    for i, node in enumerate(contents):
        path = os.path.join(root, node)
        is_last = i == len(contents) - 1

        connector = '└── ' if is_last else '├── '
        line = f'{prefix}{connector}{node}'
        tree_lines.append(line)

        if os.path.isdir(path):
            new_prefix = prefix + ('    ' if is_last else '│   ')
            tree_lines.extend(get_project_tree(path, new_prefix, ignore_list))

    return tree_lines

def update_readme_safely(tree_lines):
    readme_path = "README.md"

    if not os.path.exists(readme_path):
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# 项目说明\n\n# 项目结构\n\n")

    with open(readme_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # ====================== 安全截取 ======================
    start_idx = None
    end_idx = len(lines)

    # 1. 找到 # 项目结构
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "# 项目结构":
            start_idx = i
            break

    # 2. 找到下一个 # 标题（作为结束位置）
    if start_idx is not None:
        for i in range(start_idx + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith("# ") and stripped != "# 项目结构":
                end_idx = i
                break

    # 3. 保留：前面内容 + 项目结构标题 + 新目录树 + 后面所有内容
    new_content = []
    if start_idx is None:
        new_content = lines
        new_content.extend(["\n# 项目结构\n"])
    else:
        new_content = lines[:start_idx + 1]

    # 写入新目录树
    new_content.append("\n```\n")
    new_content.extend([line + "\n" for line in tree_lines])
    new_content.append("```\n\n")

    # 把后面原来的内容加回来（不会删除！）
    if start_idx is not None:
        new_content.extend(lines[end_idx:])

    # 保存
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.writelines(new_content)

if __name__ == "__main__":
    print("正在生成项目结构...")
    tree = get_project_tree('.')
    print("\n".join(tree))
    update_readme_safely(tree)
    print("✅ 成功更新README.md项目结构！")