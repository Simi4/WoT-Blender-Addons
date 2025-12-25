"""
Taken from:
https://github.com/blueskythlikesclouds/TagTools/blob/master/TagTools.py
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import IO, Any, Self


class TagSubType(IntFlag):
    Void = 0x0
    Invalid = 0x1
    Bool = 0x2
    String = 0x3
    Int = 0x4
    Float = 0x5
    Pointer = 0x6
    Class = 0x7
    Array = 0x8
    Tuple = 0x28
    TypeMask = 0xFF
    IsSigned = 0x200
    Float32 = 0x1746
    Int8 = 0x2000
    Int16 = 0x4000
    Int32 = 0x8000
    Int64 = 0x10000


class TagFlag(IntFlag):
    SubType = 0x1
    Pointer = 0x2
    Version = 0x4
    ByteSize = 0x8
    AbstractValue = 0x10
    Members = 0x20
    Interfaces = 0x40
    Unknown = 0x80


@dataclass
class TagMember:
    name: bytes = b''
    flags: int = 0
    byteOffset: int = 0
    typ: 'TagType | None' = None


@dataclass
class TagTemplate:
    name: bytes = b'v'
    value: Any = 0

    @property
    def isInt(self):
        return self.name[0] == b'v'

    @property
    def isType(self):
        return self.name[0] == b't'


@dataclass
class TagType:
    name: bytes = b''
    templates: list[TagTemplate] = field(default_factory=list, repr=False)
    parent: Self | None = field(default=None, repr=False)
    flags: TagFlag = TagFlag.SubType
    subTypeFlags: TagSubType = TagSubType.Void
    pointer: Self | None = None
    version: int = 0
    byteSize: int = 0
    alignment: int = 0
    abstractValue: int = 0
    members: list[TagMember] = field(default_factory=list)
    interfaces: list[tuple[Self, int]] = field(default_factory=list, repr=False)

    @property
    def superType(self):
        if not self.flags & TagFlag.SubType:
            return self.parent.superType

        else:
            return self

    @property
    def subType(self):
        return self.subTypeFlags & TagSubType.TypeMask

    @property
    def allMembers(self):
        if self.parent:
            for member in self.parent.allMembers:
                yield member

        for member in self.members:
            yield member

    @property
    def tupleSize(self):
        return self.subTypeFlags >> 8


@dataclass
class TagObject:
    value: Any
    typ: TagType = field(repr=False)


@dataclass
class TagItem:
    typ: TagType | None = None
    offset: int = 0
    count: int = 0
    isPtr: bool = False
    value: Any = None


class TagSectionReader:
    r: 'TagReader'
    offset: int
    size: int
    signature: bytes

    def __init__(self, r: 'TagReader', *signatures: bytes):
        self.r = r
        self.offset = r.f.tell() + 8
        self.size = (r.readFormat('>I') & 0x3FFFFFFF) - 8
        self.signature = r.f.read(4)

        if self.signature not in signatures:
            raise ValueError(f'Invalid signature, expected {signatures}, got {self.signature}')

    @property
    def end(self):
        return self.r.f.tell() >= (self.offset + self.size)

    def __enter__(self):
        self.r.f.seek(self.offset)
        return self

    def __exit__(self, arg1, arg2, arg3):
        self.r.f.seek(self.offset + self.size)


class TagFileType(IntEnum):
    Invalid = -1
    Object = 0
    Compendium = 1


class TagReader:
    f: IO[bytes]
    dataOffset: int
    types: list[TagType | None]
    items: list[TagItem]

    def __init__(self, f: IO[bytes]):
        self.f = f
        self.dataOffset = 0
        self.types = []
        self.items = []
        self.readRootSection()

    def __enter__(self):
        return self

    def __exit__(self, arg1, arg2, arg3):
        self.f.close()

    @staticmethod
    def fromIO(f: IO[bytes]):
        f.seek(0)
        with TagReader(f) as r:
            return r.getObject(0)

    @staticmethod
    def checkIO(f: IO[bytes]) -> TagFileType:
        f.seek(4)
        signature = f.read(4)
        if signature == b'TAG0':
            return TagFileType.Object
        elif signature == b'TCM0':
            return TagFileType.Compendium
        else:
            return TagFileType.Invalid

    @classmethod
    def fromFile(cls, inputFileName: str):
        return cls.fromIO(open(inputFileName, 'rb'))

    @classmethod
    def checkFile(cls, inputFileName: str):
        with open(inputFileName, 'rb') as f:
            return cls.checkIO(f)

    def readTypeSection(self):
        with TagSectionReader(self, b'TYPE'):
            with TagSectionReader(self, b'TST1') as t3:
                typeStrings = self.f.read(t3.size).split(b'\0')

            with TagSectionReader(self, b'TNA1'):
                typeCount = self.readPacked()
                self.types = [TagType() for x in range(typeCount)]
                self.types[0] = None

                for typ in self.types[1:]:
                    typ.name = typeStrings[self.readPacked()]

                    for i in range(self.readPacked()):
                        template = TagTemplate(typeStrings[self.readPacked()], self.readPacked())

                        if template.isType:
                            template.value = self.types[template.value]

                        typ.templates.append(template)

            with TagSectionReader(self, b'FST1') as t5:
                fieldStrings = self.f.read(t5.size).split(b'\0')

            with TagSectionReader(self, b'TBDY') as t6:
                while not t6.end:
                    typeIndex = self.readPacked()

                    if typeIndex == 0:
                        continue

                    typ = self.types[typeIndex]
                    typ.parent = self.types[self.readPacked()]
                    typ.flags = TagFlag(self.readPacked())

                    if typ.flags & TagFlag.SubType:
                        typ.subTypeFlags = TagSubType(self.readPacked())

                    if typ.flags & TagFlag.Pointer:
                        typ.pointer = self.types[self.readPacked()]

                    if typ.flags & TagFlag.Version:
                        typ.version = self.readPacked()

                    if typ.flags & TagFlag.ByteSize:
                        typ.byteSize = self.readPacked()
                        typ.alignment = self.readPacked()

                    if typ.flags & TagFlag.AbstractValue:
                        typ.abstractValue = self.readPacked()

                    if typ.flags & TagFlag.Members:
                        members_len = self.readPacked()
                        unknown = members_len >> 6
                        assert unknown in [0, 1024, 2048]
                        members_len = members_len & 0x3F
                        for i in range(members_len):
                            member = TagMember()
                            member.name = fieldStrings[self.readPacked()]
                            member.flags = self.readPacked()
                            member.byteOffset = self.readPacked()
                            typ_index = self.readPacked(True)
                            member.typ = self.types[typ_index]
                            typ.members.append(member)

                    if typ.flags & TagFlag.Interfaces:
                        typ.interfaces = [(self.types[self.readPacked()], self.readPacked()) for x in range(self.readPacked())]

                    if typ.flags & TagFlag.Unknown:
                        raise ValueError('Flag 0x80 exists, handle it!')

            with TagSectionReader(self, b'TPAD'):
                pass

    def readIndexSection(self):
        with TagSectionReader(self, b'INDX'):
            with TagSectionReader(self, b'ITEM') as t2:
                while not t2.end:
                    item = TagItem()
                    flag = self.readFormat('<I')
                    item.typ = self.types[flag & 0xFFFFFF]
                    item.isPtr = bool(flag & 0x10000000)
                    item.offset = self.dataOffset + self.readFormat('<I')
                    item.count = self.readFormat('<I')
                    self.items.append(item)

    def readRootSection(self):
        with TagSectionReader(self, b'TAG0'):
            with TagSectionReader(self, b'SDKV') as t2:
                version = self.f.read(t2.size)
                if version != b'20200200':
                    raise ValueError('Invalid SDK version.')

            with TagSectionReader(self, b'DATA') as t3:
                self.dataOffset = t3.offset

            self.readTypeSection()
            self.readIndexSection()

    @staticmethod
    def getFormatString(flags, signed=False):
        ret = ''

        if flags & TagSubType.Int8:
            ret = 'B'

        elif flags & TagSubType.Int16:
            ret = '<H'

        elif flags & TagSubType.Int32:
            ret = '<I'

        elif flags & TagSubType.Int64:
            ret = '<Q'

        if flags & TagSubType.IsSigned or signed:
            return ret.lower()

        else:
            return ret

    def readObject(self, typ: TagType, offset: int = 0) -> TagObject:
        if offset == 0:
            offset = self.f.tell()

        else:
            self.f.seek(offset)

        typOrg = typ
        typ = typ.superType

        value = None

        if typ.subType == TagSubType.Bool:
            value = self.readFormat(TagReader.getFormatString(typ.subTypeFlags)) > 0

        elif typ.subType == TagSubType.String:
            value = ''.join(map(chr, [x.value for x in self.readItemPtr()[:-1]]))

        elif typ.subType == TagSubType.Int:
            value = self.readFormat(TagReader.getFormatString(typ.subTypeFlags))

        elif typ.subType == TagSubType.Float:
            value = self.readFormat('<f')

        elif typ.subType == TagSubType.Pointer:
            value = self.readItemPtr()

            if len(value) == 1:
                value = value[0]

            else:
                value = None

        elif typ.subType == TagSubType.Class:
            value = {x.name: self.readObject(x.typ, offset + x.byteOffset) for x in typ.allMembers}

        elif typ.subType == TagSubType.Array:
            value = self.readItemPtr()

        elif typ.subType == TagSubType.Tuple:
            value = tuple([self.readObject(typ.pointer, offset + x * typ.pointer.superType.byteSize) for x in range(typ.tupleSize)])

        # hack!!!
        elif typ.subType == 131 and typ.byteSize == 4:
            value = self.readItemPtr()
            value = ''.join(list(map(lambda x: chr(x.value), value))[:-1])

        self.f.seek(offset + typ.byteSize)
        return TagObject(value, typOrg)

    def readItemPtr(self):
        index = self.readFormat('<I')

        if index == 0:
            return []

        else:
            item = self.items[index]

            if item.value is None:
                item.value = [self.readObject(item.typ, item.offset + x * item.typ.superType.byteSize) for x in range(item.count)]

            return item.value

    def readFormat(self, format):
        data = struct.unpack(format, self.f.read(struct.calcsize(format)))

        if len(data) == 1:
            return data[0]

        else:
            return data

    def readPacked(self, debug: bool = False) -> int:
        """
        https://github.com/blueskythlikesclouds/TagTools/blob/havoc/Havoc/Extensions/BinaryReaderEx.cs
        """
        firstByte = self.readFormat('B')

        if firstByte & 0x80 == 0:
            return firstByte

        match firstByte >> 3:
            case num if 0x10 <= num <= 0x17:
                return ((firstByte << 8) | self.readFormat('B')) & 0x3FFF
            case num if 0x18 <= num <= 0x1B:
                return ((firstByte << 16) | self.readFormat('>H')) & 0x1FFFFF
            case 0x1C:
                return ((firstByte << 24) | (self.readFormat('B') << 16) | self.readFormat('>H')) & 0x7FFFFFF
            case 0x1D:
                return self.readFormat('>I')
            case 0x1E:
                # TODO
                assert False
            case 0x1F:
                # TODO
                assert False
            case _:
                return 0

    def getObject(self, index: int) -> TagObject:
        item = self.items[index + 1]

        if item.typ is None:
            return None

        if item.value is None:
            item.value = [self.readObject(item.typ, item.offset + x * item.typ.superType.byteSize) for x in range(item.count)]

        return item.value[0]
