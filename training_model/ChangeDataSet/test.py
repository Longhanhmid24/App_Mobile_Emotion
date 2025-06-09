import subprocess

def get_large_files(size_limit_mb=100):
    size_limit_bytes = size_limit_mb * 1024 * 1024
    cmd = ['git', 'rev-list', '--objects', '--all']
    output = subprocess.check_output(cmd, text=True)

    objects = [line.strip().split() for line in output.strip().split('\n') if len(line.strip().split()) == 2]
    for sha, path in objects:
        try:
            size_output = subprocess.check_output(['git', 'cat-file', '-s', sha], text=True)
            size = int(size_output.strip())
            if size > size_limit_bytes:
                print(f"{size / (1024 * 1024):.2f} MB\t{path}")
        except subprocess.CalledProcessError:
            continue

if __name__ == "__main__":
    get_large_files(100)  # Change 100 to other MB size if needed
