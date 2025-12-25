


__all__ = ('ResourceManager',)


from io import BytesIO
from xml.etree.ElementTree import Element
from zipfile import ZipFile, ZipInfo
from .common import *


pkg_filename_list = (
    'vehicles_level_01_hd.pkg', 'vehicles_level_01.pkg',
    'vehicles_level_02_hd.pkg', 'vehicles_level_02.pkg',
    'vehicles_level_03_hd.pkg', 'vehicles_level_03.pkg',
    'vehicles_level_04_hd.pkg', 'vehicles_level_04.pkg',

    'vehicles_level_05_hd.pkg', 'vehicles_level_05.pkg',
    'vehicles_level_05_hd-part1.pkg', 'vehicles_level_05_hd-part2.pkg',
    'vehicles_level_05-part1.pkg', 'vehicles_level_05-part2.pkg',

    'vehicles_level_06_hd.pkg', 'vehicles_level_06.pkg',
    'vehicles_level_06_hd-part1.pkg', 'vehicles_level_06_hd-part2.pkg',
    'vehicles_level_06-part1.pkg', 'vehicles_level_06-part2.pkg',

    'vehicles_level_07_hd.pkg', 'vehicles_level_07.pkg',
    'vehicles_level_07_hd-part1.pkg', 'vehicles_level_07_hd-part2.pkg',
    'vehicles_level_07-part1.pkg', 'vehicles_level_07-part2.pkg',

    'vehicles_level_08_hd.pkg', 'vehicles_level_08.pkg',
    'vehicles_level_08_hd-part1.pkg', 'vehicles_level_08_hd-part2.pkg', 'vehicles_level_08_hd-part3.pkg',
    'vehicles_level_08-part1.pkg', 'vehicles_level_08-part2.pkg', 'vehicles_level_08-part3.pkg',

    'vehicles_level_09_hd.pkg', 'vehicles_level_09.pkg',
    'vehicles_level_09_hd-part1.pkg', 'vehicles_level_09_hd-part2.pkg',
    'vehicles_level_09-part1.pkg', 'vehicles_level_09-part2.pkg',

    'vehicles_level_10_hd.pkg', 'vehicles_level_10.pkg',
    'vehicles_level_10_hd-part1.pkg', 'vehicles_level_10_hd-part2.pkg', 'vehicles_level_10_hd-part3.pkg',
    'vehicles_level_10-part1.pkg', 'vehicles_level_10-part2.pkg', 'vehicles_level_10-part3.pkg',

    'vehicles_level_11_hd.pkg', 'vehicles_level_11.pkg',
    'vehicles_level_11_hd-part1.pkg', 'vehicles_level_11_hd-part2.pkg',
    'vehicles_level_11-part1.pkg', 'vehicles_level_11-part2.pkg',

    'shared_content_hd.pkg', 'shared_content.pkg',
    'shared_content_hd-part1.pkg', 'shared_content_hd-part2.pkg',
    'shared_content-part1.pkg', 'shared_content-part2.pkg', 'shared_content-part3.pkg',

    'shared_content_sandbox_hd.pkg', 'shared_content_sandbox.pkg',
    'shared_content_sandbox_hd-part1.pkg', 'shared_content_sandbox_hd-part2.pkg',
    'shared_content_sandbox-part1.pkg', 'shared_content_sandbox-part2.pkg',

    'particles.pkg',
)



class ResourceManager:
    res_path: Path
    scripts_pkg: Optional[ZipFile]
    pkg_filepath_dict: dict[str, tuple[ZipFile, ZipInfo]]

    def __init__(self, res_path: str):
        self.res_path = Path(res_path)
        self.scripts_pkg = None
        self.pkg_filepath_dict = {}

        scripts_pkg_path = self.res_path / 'packages' / 'scripts.pkg'
        if scripts_pkg_path.is_file():
            self.scripts_pkg = ZipFile(scripts_pkg_path, 'r')

        for pkg in pkg_filename_list:
            pkg_abs_path = self.res_path / 'packages' / pkg
            if pkg_abs_path.is_file():
                zfile = ZipFile(pkg_abs_path, 'r')
                for member in zfile.filelist:
                    if member.is_dir():
                        continue
                    if member.filename.endswith(('.visual', '.visual_processed', '.primitives', '.primitives_processed', '.dds', '.track', '.model', '.havok')):
                        self.pkg_filepath_dict[member.filename.lower()] = (zfile, member)

    def exists(self, filename: str) -> bool:
        return filename.lower() in self.pkg_filepath_dict

    def open_file(self, filename: str) -> Optional[BytesIO]:
        filename = filename.lower()
        if item := self.pkg_filepath_dict.get(filename):
            return item[0].open(item[1], 'r')
        return None

    def open_scripts_xml(self, filename: Path) -> Optional[Element]:
        if self.scripts_pkg is None:
            xml_path = self.res_path / filename
            if not xml_path.is_file():
                return
            with open(xml_path, 'rb') as f:
                element = g_XmlUnpacker.read(f)
        else:
            with self.scripts_pkg.open(filename.as_posix(), 'r') as f:
                element = g_XmlUnpacker.read(f)
        return element
