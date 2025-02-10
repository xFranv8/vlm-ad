import os


root = "data"
base_paths: list[str] = list(sorted(os.listdir(root)))

for base_path in base_paths:
    route_town_path: list[str] = list(sorted(os.listdir(os.path.join(root, base_path))))
    for path in route_town_path:
        if not os.path.exists(os.path.join(root, base_path, path, "measurements")):
            continue

        steps: list[str] = list(sorted(os.listdir(os.path.join(root, base_path, path, "measurements"))))
        for step in steps:
            full_path = os.path.join(root, base_path, path, "measurements", step)

            if full_path.endswith(".json"):
                continue

            # unzip the file and remove the zip file
            os.system(f"gzip -d {full_path}")
            # os.system(f"rm {full_path}")