from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Paths:
    model_load_path: Optional[Path]
    model_save_path: Path
    output_path: Path
    report_file: Path
    rank_file: Path

    def with_split(self, split_id: int, params) -> "Paths":
        """Return a new Paths object adjusted for a particular split ID.

        This mirrors the old behavior of swapping module-level globals but keeps
        things immutable and local.
        """
        # derive new locations
        model_save_path = self.model_save_path / f"{split_id}"
        output_path = self.output_path / f"{split_id}"

        report_parent = self.report_file.parent / f"{split_id}"
        report_parent.mkdir(parents=True, exist_ok=True)
        report_file = report_parent / self.report_file.name

        rank_file = self.rank_file
        if params.mode == "fine-tune":
            rank_parent = self.rank_file.parent / f"{split_id}"
            rank_parent.mkdir(parents=True, exist_ok=True)
            rank_file = rank_parent / self.rank_file.name

        model_load_path = self.model_load_path
        if params.mode == "interpret" and model_load_path is not None:
            model_load_path = model_load_path / f"{split_id}"

        return Paths(
            model_load_path=model_load_path,
            model_save_path=model_save_path,
            output_path=output_path,
            report_file=report_file,
            rank_file=rank_file,
        )
