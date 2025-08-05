#!/usr/bin/env python3
"""
Export OpenAPI specification from the Dx0 FastAPI application.

This script generates JSON and YAML versions of the OpenAPI specification
that can be used with external tools like Swagger UI, Postman, or code generators.
"""

import json
import sys
import yaml
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sdb.ui.app import app


def export_openapi_spec(output_dir: Path = None, formats: list = None):
    """
    Export the OpenAPI specification in JSON and YAML formats.
    
    Args:
        output_dir: Directory to save the files (defaults to docs/)
        formats: List of formats to export ('json', 'yaml', or both)
    """
    if output_dir is None:
        output_dir = project_root / "docs"
    
    if formats is None:
        formats = ["json", "yaml"]
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Get the OpenAPI specification from the FastAPI app
    openapi_spec = app.openapi()
    
    # Export JSON format
    if "json" in formats:
        json_path = output_dir / "openapi.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(openapi_spec, f, indent=2, ensure_ascii=False)
        print(f"✅ Exported OpenAPI JSON to: {json_path}")
    
    # Export YAML format
    if "yaml" in formats:
        yaml_path = output_dir / "openapi.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(openapi_spec, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ Exported OpenAPI YAML to: {yaml_path}")
    
    return openapi_spec


def print_openapi_info(spec: dict):
    """Print information about the OpenAPI specification."""
    info = spec.get("info", {})
    paths = spec.get("paths", {})
    components = spec.get("components", {})
    
    print("\n📋 OpenAPI Specification Summary:")
    print(f"   Title: {info.get('title', 'N/A')}")
    print(f"   Version: {info.get('version', 'N/A')}")
    print(f"   Description: {info.get('description', 'N/A')[:100]}...")
    print(f"   Endpoints: {len(paths)}")
    print(f"   Schemas: {len(components.get('schemas', {}))}")
    print(f"   Security Schemes: {len(components.get('securitySchemes', {}))}")
    
    print("\n🔗 Available Endpoints:")
    for path, methods in paths.items():
        for method, details in methods.items():
            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                summary = details.get("summary", "No summary")
                print(f"   {method.upper():6} {path:30} - {summary}")


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export OpenAPI specification from Dx0 API")
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=None,
        help="Output directory (defaults to docs/)"
    )
    parser.add_argument(
        "--format", 
        choices=["json", "yaml", "both"], 
        default="both",
        help="Export format (default: both)"
    )
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Print information about the OpenAPI spec"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate the OpenAPI specification"
    )
    
    args = parser.parse_args()
    
    # Determine formats to export
    if args.format == "both":
        formats = ["json", "yaml"]
    else:
        formats = [args.format]
    
    try:
        # Export the specification
        spec = export_openapi_spec(args.output_dir, formats)
        
        # Print information if requested
        if args.info:
            print_openapi_info(spec)
        
        # Validate if requested
        if args.validate:
            validate_openapi_spec(spec)
            
    except Exception as e:
        print(f"❌ Error exporting OpenAPI specification: {e}")
        sys.exit(1)


def validate_openapi_spec(spec: dict):
    """
    Validate the OpenAPI specification for common issues.
    
    Args:
        spec: The OpenAPI specification dictionary
    """
    issues = []
    
    # Check required fields
    if "openapi" not in spec:
        issues.append("Missing 'openapi' version field")
    
    if "info" not in spec:
        issues.append("Missing 'info' section")
    else:
        info = spec["info"]
        if "title" not in info:
            issues.append("Missing title in info section")
        if "version" not in info:
            issues.append("Missing version in info section")
    
    # Check paths
    paths = spec.get("paths", {})
    if not paths:
        issues.append("No paths defined")
    
    # Check for missing descriptions
    missing_descriptions = []
    for path, methods in paths.items():
        for method, details in methods.items():
            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                if not details.get("description") and not details.get("summary"):
                    missing_descriptions.append(f"{method.upper()} {path}")
    
    if missing_descriptions:
        issues.append(f"Missing descriptions for: {', '.join(missing_descriptions[:5])}")
        if len(missing_descriptions) > 5:
            issues.append(f"... and {len(missing_descriptions) - 5} more endpoints")
    
    # Check security schemes
    components = spec.get("components", {})
    security_schemes = components.get("securitySchemes", {})
    if not security_schemes:
        issues.append("No security schemes defined")
    
    # Print validation results
    if issues:
        print("\n⚠️  OpenAPI Specification Issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("\n✅ OpenAPI specification validation passed")


def create_swagger_ui_html(output_dir: Path = None):
    """
    Create a simple Swagger UI HTML file that can be served locally.
    
    Args:
        output_dir: Directory to save the HTML file
    """
    if output_dir is None:
        output_dir = project_root / "docs"
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Dx0 API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html {
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }
        *, *:before, *:after {
            box-sizing: inherit;
        }
        body {
            margin:0;
            background: #fafafa;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: './openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            });
        };
    </script>
</body>
</html>"""
    
    html_path = output_dir / "swagger-ui.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ Created Swagger UI HTML at: {html_path}")
    print(f"   Open in browser: file://{html_path.absolute()}")


if __name__ == "__main__":
    main()