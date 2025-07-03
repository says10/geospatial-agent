import sys
import subprocess
import platform

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None

class HealthChecker:
    def __init__(self):
        self.results = {}

    def _run_check(self, name, function):
        try:
            result = function()
            self.results[name] = {"status": "✅ PASS", "details": str(result)}
        except Exception as e:
            self.results[name] = {"status": "❌ FAIL", "details": str(e).strip()}

    def check_python_version(self):
        version = platform.python_version()
        if not version.startswith("3.10"):
            raise ValueError(f"Expected Python ~3.10, but found {version}")
        return f"Python {version}"

    def check_gdal_version(self):
        result = subprocess.run(
            ["gdalinfo", "--version"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    def check_gis_libs_import(self):
        import geopandas, rasterio, shapely
        return (
            f"GeoPandas {geopandas.__version__}, "
            f"Rasterio {rasterio.__version__}, "
            f"Shapely {shapely.__version__}"
        )

    def report(self):
        title = "🚀 Geospatial Agent (Lite) Environment Healthcheck 🚀"
        if console:
            console.print(f"[bold cyan]{title}[/bold cyan]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Check", style="dim", width=30)
            table.add_column("Status", justify="center")
            table.add_column("Details")
            for name, result in self.results.items():
                table.add_row(name, result["status"], result["details"])
            console.print(table)
        else:
            # Fallback for non-rich console
            print(title)
            for name, result in self.results.items():
                print(f"- {name:<30} | {result['status']:<10} | {result['details']}")

        if any(res["status"] == "❌ FAIL" for res in self.results.values()):
            print("\n🚨 One or more healthchecks failed.")
            sys.exit(1)
        else:
            print("\n🎉 All checks passed. The lean foundation is ready!")

    def run_all(self):
        self._run_check("Python Version", self.check_python_version)
        self._run_check("Core Python GIS Libs", self.check_gis_libs_import)
        self._run_check("GDAL Command Line Tool", self.check_gdal_version)
        self.report()

if __name__ == "__main__":
    checker = HealthChecker()
    checker.run_all()