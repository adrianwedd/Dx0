#!/usr/bin/env python3
"""
Migrate tasks from tasks.yml to GitHub issues with duplicate prevention and label management.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install PyYAML")
    sys.exit(1)


def check_prerequisites():
    """Check if GitHub CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        if result.returncode != 0:
            print("Error: GitHub CLI is not authenticated. Run: gh auth login")
            return False
        return True
    except FileNotFoundError:
        print("Error: GitHub CLI (gh) is not installed. Install from: https://cli.github.com/")
        return False


def get_existing_issues() -> Set[str]:
    """Get titles of existing GitHub issues to prevent duplicates."""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--limit", "200", "--json", "title"],
            capture_output=True,
            text=True,
            check=True
        )
        issues = json.loads(result.stdout)
        return {issue["title"] for issue in issues}
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not fetch existing issues: {e}")
        return set()
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse existing issues: {e}")
        return set()


def get_existing_labels() -> Set[str]:
    """Get existing GitHub labels."""
    try:
        result = subprocess.run(
            ["gh", "label", "list", "--json", "name"],
            capture_output=True,
            text=True,
            check=True
        )
        labels = json.loads(result.stdout)
        return {label["name"] for label in labels}
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not fetch existing labels: {e}")
        return set()
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse existing labels: {e}")
        return set()


def create_label(name: str, color: str, description: str = "", dry_run: bool = False) -> bool:
    """Create a GitHub label."""
    if dry_run:
        print(f"  [DRY RUN] Would create label: {name} (#{color}) - {description}")
        return True
    
    try:
        cmd = ["gh", "label", "create", name, "--color", color]
        if description:
            cmd.extend(["--description", description])
        
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✓ Created label: {name}")
        return True
    except subprocess.CalledProcessError as e:
        # Label might already exist, which is fine
        if "already exists" in e.stderr:
            print(f"  - Label already exists: {name}")
            return True
        print(f"  ✗ Failed to create label {name}: {e.stderr}")
        return False


def setup_labels(all_labels: Set[str], dry_run: bool = False) -> bool:
    """Create all necessary labels with appropriate colors."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Setting up GitHub labels...")
    
    existing_labels = get_existing_labels()
    new_labels = all_labels - existing_labels
    
    if not new_labels:
        print("  All labels already exist!")
        return True
    
    # Define label colors and descriptions
    label_config = {
        # Component labels
        "backend": ("0052cc", "Backend/server-side changes"),
        "frontend": ("1d76db", "Frontend/UI changes"),
        "cli": ("000000", "Command-line interface"),
        "data": ("f9d0c4", "Data processing or storage"),
        "ai": ("5319e7", "AI/ML functionality"),
        "ml": ("5319e7", "Machine learning"),
        
        # Type labels
        "feature": ("a2eeef", "New feature or functionality"),
        "enhancement": ("84b6eb", "Enhancement to existing feature"),
        "bug": ("d73a4a", "Bug fix"),
        "refactor": ("fbca04", "Code refactoring"),
        "test": ("0e8a16", "Testing related"),
        "documentation": ("0075ca", "Documentation"),
        
        # Priority labels
        "high": ("b60205", "High priority"),
        "medium": ("fbca04", "Medium priority"),
        "low": ("0e8a16", "Low priority"),
        
        # Phase labels
        "phase-0": ("e6e6fa", "Phase 0: Engineering Foundation"),
        "phase-1": ("dda0dd", "Phase 1: Core Features"),
        "phase-2": ("d8bfd8", "Phase 2: Advanced Features"),
        "phase-3": ("daa520", "Phase 3: Optimization"),
        
        # Process labels
        "ci": ("f0f8ff", "Continuous Integration"),
        "devops": ("2f4f4f", "DevOps and deployment"),
        "tooling": ("708090", "Development tooling"),
        "quality": ("32cd32", "Code quality"),
        "observability": ("ff6347", "Monitoring and observability"),
        "project-management": ("deb887", "Project management"),
        
        # Migration marker
        "migrated-from-yaml": ("fef2c0", "Migrated from tasks.yml"),
    }
    
    success = True
    for label in new_labels:
        color, description = label_config.get(label, ("ededed", ""))
        if not create_label(label, color, description, dry_run):
            success = False
    
    return success


def format_task_description(task: Dict[str, Any], phase_name: str = "") -> str:
    """Format a task description for GitHub issue."""
    lines = []
    
    if phase_name:
        lines.append(f"**Phase**: {phase_name}")
        lines.append("")
    
    # Main description
    if "description" in task:
        lines.append(task["description"].strip())
        lines.append("")
    
    # Add component/area info if available
    if "component" in task:
        lines.append(f"**Component**: {task['component']}")
    if "area" in task:
        lines.append(f"**Area**: {task['area']}")
    if "priority" in task:
        lines.append(f"**Priority**: {task['priority']}")
    
    # Add actionable steps if available
    if "actionable_steps" in task:
        lines.append("")
        lines.append("## Actionable Steps")
        for i, step in enumerate(task["actionable_steps"], 1):
            lines.append(f"{i}. {step}")
    
    # Add acceptance criteria if available
    if "acceptance_criteria" in task:
        lines.append("")
        lines.append("## Acceptance Criteria")
        for criterion in task["acceptance_criteria"]:
            lines.append(f"- {criterion}")
    
    # Add subtasks if available
    if "subtasks" in task:
        lines.append("")
        lines.append("## Subtasks")
        for subtask in task["subtasks"]:
            status = "x" if subtask.get("done", False) else " "
            lines.append(f"- [{status}] {subtask['title']}")
    
    # Add migration marker
    lines.append("")
    lines.append("---")
    lines.append("*Migrated from tasks.yml*")
    
    return "\n".join(lines)


def collect_all_labels(tasks_data: Dict[str, Any]) -> Set[str]:
    """Collect all labels that will be used."""
    all_labels = set()
    
    # Process phase tasks
    for phase in tasks_data.get("phases", []):
        phase_name = phase.get("name", "")
        phase_num = None
        if "Phase 0" in phase_name:
            phase_num = 0
        elif "Phase 1" in phase_name:
            phase_num = 1
        elif "Phase 2" in phase_name:
            phase_num = 2
        elif "Phase 3" in phase_name:
            phase_num = 3
        
        if phase_num is not None:
            all_labels.add(f"phase-{phase_num}")
        
        for task in phase.get("tasks", []):
            # Original labels
            if "labels" in task:
                all_labels.update(str(label) for label in task["labels"])
    
    # Process backlog tasks
    for task in tasks_data.get("backlog", []):
        if "labels" in task:
            all_labels.update(str(label) for label in task["labels"])
        if "component" in task:
            all_labels.add(str(task["component"]))
        if "area" in task:
            all_labels.add(str(task["area"]))
        if "priority" in task:
            all_labels.add(str(task["priority"]))
    
    # Add migration marker
    all_labels.add("migrated-from-yaml")
    
    return all_labels


def create_github_issue(title: str, description: str, labels: List[str], dry_run: bool = False) -> bool:
    """Create a GitHub issue."""
    if dry_run:
        print(f"  [DRY RUN] Would create issue: {title}")
        print(f"    Labels: {', '.join(labels)}")
        return True
    
    try:
        cmd = ["gh", "issue", "create", "--title", title, "--body", description]
        if labels:
            cmd.extend(["--label", ",".join(labels)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        issue_url = result.stdout.strip()
        print(f"  ✓ Created: {title}")
        print(f"    URL: {issue_url}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to create issue '{title}': {e.stderr}")
        return False


def process_tasks(file_path: Path, dry_run: bool = False):
    """Process tasks from YAML file and create GitHub issues."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tasks_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    if not check_prerequisites():
        return False
    
    # Collect all labels first
    all_labels = collect_all_labels(tasks_data)
    print(f"Found {len(all_labels)} unique labels to set up")
    
    # Setup labels
    if not setup_labels(all_labels, dry_run):
        print("Warning: Some labels could not be created")
    
    # Get existing issues to prevent duplicates
    existing_issues = get_existing_issues()
    print(f"\nFound {len(existing_issues)} existing GitHub issues")
    
    created_count = 0
    skipped_count = 0
    duplicate_count = 0
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing tasks...")
    
    # Process phase tasks
    for phase in tasks_data.get("phases", []):
        phase_name = phase.get("name", "")
        print(f"\nProcessing {phase_name}:")
        
        phase_num = None
        if "Phase 0" in phase_name:
            phase_num = 0
        elif "Phase 1" in phase_name:
            phase_num = 1
        elif "Phase 2" in phase_name:
            phase_num = 2
        elif "Phase 3" in phase_name:
            phase_num = 3
        
        for task in phase.get("tasks", []):
            if task.get("done", False):
                skipped_count += 1
                continue
            
            title = task.get("title", f"Task {task.get('id', 'Unknown')}")
            
            # Check for duplicates
            if title in existing_issues:
                print(f"  - Duplicate found, skipping: {title}")
                duplicate_count += 1
                continue
            
            description = format_task_description(task, phase_name)
            
            # Prepare labels
            labels = [str(label) for label in task.get("labels", [])]
            if phase_num is not None:
                labels.append(f"phase-{phase_num}")
            labels.append("migrated-from-yaml")
            
            if create_github_issue(title, description, labels, dry_run):
                created_count += 1
    
    # Process backlog tasks
    backlog = tasks_data.get("backlog", [])
    if backlog:
        print(f"\nProcessing backlog ({len(backlog)} items):")
        
        for task in backlog:
            if task.get("status") == "done":
                skipped_count += 1
                continue
            
            title = task.get("title", "Untitled Task")
            
            # Check for duplicates
            if title in existing_issues:
                print(f"  - Duplicate found, skipping: {title}")
                duplicate_count += 1
                continue
            
            description = format_task_description(task)
            
            # Prepare labels
            labels = [str(label) for label in task.get("labels", [])]
            if "component" in task:
                labels.append(str(task["component"]))
            if "area" in task:
                labels.append(str(task["area"]))
            if "priority" in task:
                labels.append(str(task["priority"]))
            labels.append("migrated-from-yaml")
            
            if create_github_issue(title, description, labels, dry_run):
                created_count += 1
    
    # Summary
    print(f"\n{'DRY RUN ' if dry_run else ''}Summary:")
    print(f"  Issues created: {created_count}")
    print(f"  Tasks skipped (completed): {skipped_count}")
    print(f"  Duplicates skipped: {duplicate_count}")
    
    if dry_run:
        print(f"\nRun without --dry-run to actually create {created_count} GitHub issues")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate tasks from YAML to GitHub issues")
    parser.add_argument(
        "--file", 
        type=Path, 
        default=Path("tasks.yml"),
        help="Path to tasks YAML file (default: tasks.yml)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without actually creating issues"
    )
    
    args = parser.parse_args()
    
    if not args.file.exists():
        print(f"Error: File {args.file} not found")
        return 1
    
    success = process_tasks(args.file, args.dry_run)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())