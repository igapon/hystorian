import gc
import os
import pickle
import struct
from pathlib import Path

import numpy as np

from .utils import HyConvertedData, conversion_metadata

"""
Following code was translated from the original Matlab source code written by Matthew Poss available here: https://ch.mathworks.com/matlabcentral/fileexchange/80212-ardf-to-matlab/?s_tid=LandingPageTabfx
"""


def parseNotes(_nts):
    ret = []
    for i in range(len(_nts)):
        nts = _nts[i]
        ntsStruct = {}

        siz = len(nts)
        idx = 0

        pntTitleStart = 0
        pntTitleEnd = 0
        pntFirstColon = 0
        pntDataStart = 0
        pntDataEnd = 0

        found = 0

        while idx < siz:
            if (nts[idx] == ord(":")) and (found == 0):
                pntFirstColon = idx
                pntTitleEnd = idx
                if nts[idx + 1] == ord(" "):
                    pntDataStart = idx + 2
                else:
                    pntDataStart = idx + 1

                found = 1

            if nts[idx] == ord("\r"):
                pntDataEnd = idx - 1

                tempStr = nts[pntTitleStart:pntTitleEnd]
                tempStr.strip()
                tempStr = tempStr.replace(b".", b"")
                titleSize = len(tempStr)

                dataSize = pntDataEnd - pntDataStart + 1
                tempAr = nts[pntDataStart : pntDataEnd + 1]

                # print(tempStr)
                if found == 1 and len(tempStr) > 0:
                    if (tempStr[0] >= ord("0")) and (tempStr[0] <= ord("9")):
                        tempStr = tempStr.decode("ascii", "replace")
                        tempAr = tempAr.decode("ascii", "replace")
                        ntsStruct["n" + tempStr] = tempAr
                    else:
                        tempStr = tempStr.decode("ascii", "replace")
                        tempAr = tempAr.decode("ascii", "replace")
                        ntsStruct[tempStr] = tempAr

                pntTitleStart = idx + 1
                found = 0

            idx = idx + 1
        ret.append(ntsStruct)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def local_readTOC(fid, address, _type):
    nullCase = b"\0\0\0\0"

    if address != -1:
        fid.seek(address, 0)

    [dumCRC, lastSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)
    local_checkType(lastType, _type, fid)

    toc = {}
    toc["sizeTable"] = struct.unpack("Q", fid.read(8))[0]
    toc["numbEntry"] = struct.unpack("I", fid.read(4))[0]
    toc["sizeEntry"] = struct.unpack("I", fid.read(4))[0]

    if toc["sizeEntry"] == 24:
        # FTOC, IMAG, VOLM
        toc["pntImag"] = []
        toc["pntVolm"] = []
        toc["pntNext"] = []
        toc["pntNset"] = []
        toc["pntThmb"] = []
    elif toc["sizeEntry"] == 32:
        # TTOC
        toc["idxText"] = []
        toc["pntText"] = []
    elif toc["sizeEntry"] == 40:
        # VOFF
        toc["pntCounter"] = []
        toc["linCounter"] = []
        toc["linPointer"] = []
    else:
        # IDAT
        toc["data"] = []
        sizeRead = int((toc["sizeEntry"] - 16) / 4)

    done = 0
    numbRead = 1
    while (done == 0) and (numbRead <= toc["numbEntry"]):
        [dumCRC, dumSize, typeEntry, dumMisc] = local_readARDFpointer(fid, -1)

        if toc["sizeEntry"] == 24:
            # FTOC, IMAG, VOLM
            lastPointer = struct.unpack("Q", fid.read(8))[0]
        elif toc["sizeEntry"] == 32:
            # TTOC
            lastIndex = struct.unpack("Q", fid.read(8))[0]
            lastPointer = struct.unpack("Q", fid.read(8))[0]
        elif toc["sizeEntry"] == 40:
            # VOFF
            lastPntCount = struct.unpack("I", fid.read(4))[0]
            lastLinCount = struct.unpack("I", fid.read(4))[0]
            dum = struct.unpack("Q", fid.read(8))[0]
            lastLinPoint = struct.unpack("Q", fid.read(8))[0]
        else:
            # IDAT
            lastData = struct.unpack(str(sizeRead) + "i", fid.read(4 * sizeRead))

        if typeEntry == b"IMAG":
            toc["pntImag"].append(lastPointer)
        elif typeEntry == b"VOLM":
            toc["pntVolm"].append(lastPointer)
        elif typeEntry == b"NEXT":
            toc["pntNext"].append(lastPointer)
        elif typeEntry == b"NSET":
            toc["pntNset"].append(lastPointer)
        elif typeEntry == b"THMB":
            toc["pntThmb"].append(lastPointer)
        elif typeEntry == b"TOFF":
            toc["idxText"].append(lastIndex)
            toc["pntText"].append(lastPointer)
        elif typeEntry == b"IDAT":
            toc["data"].append(lastData)
        elif typeEntry == b"VOFF":
            toc["pntCounter"].append(lastPntCount)
            toc["linCounter"].append(lastLinCount)
            toc["linPointer"].append(lastLinPoint)
        elif typeEntry == nullCase:
            if lastType == b"IBOX":
                toc["data"].append(lastData)
            elif lastType == b"VTOC":
                toc["pntCounter"].append(lastPntCount)
                toc["linCounter"].append(lastLinCount)
                toc["linPointer"].append(lastLinPoint)
            else:
                done = 1
        else:
            print("ERROR: " + str(typeEntry) + "not recognized!")

        numbRead = numbRead + 1
    return toc


def local_readVSET(fid, address):
    if address != -1:
        fid.seek(address, 0)

    [dumCRC, lastSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)
    local_checkType(lastType, b"VSET", fid)

    vset = {}
    vset["force"] = struct.unpack("I", fid.read(4))[0]
    vset["line"] = struct.unpack("I", fid.read(4))[0]
    vset["point"] = struct.unpack("I", fid.read(4))[0]
    dum = struct.unpack("I", fid.read(4))[0]
    vset["prev"] = struct.unpack("Q", fid.read(8))[0]
    vset["next"] = struct.unpack("Q", fid.read(8))[0]

    return vset


def local_readVNAM(fid, address):
    if address != -1:
        fid.seek(address, 0)

    [dumCRC, lastSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)
    local_checkType(lastType, b"VNAM", fid)

    vnam = {}
    vnam["force"] = struct.unpack("I", fid.read(4))[0]
    vnam["line"] = struct.unpack("I", fid.read(4))[0]
    vnam["point"] = struct.unpack("I", fid.read(4))[0]
    vnam["sizeText"] = struct.unpack("I", fid.read(4))[0]
    vnam["name"] = struct.unpack(str(vnam["sizeText"]) + "s", fid.read(vnam["sizeText"]))[0]

    remainingSize = lastSize - 16 - vnam["sizeText"] - 16

    dum = struct.unpack(str(remainingSize) + "s", fid.read(remainingSize))[0]

    return vnam


def local_readVDAT(fid, address):
    if address != -1:
        fid.seek(address, 0)

    [dumCRC, lastSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)
    local_checkType(lastType, b"VDAT", fid)

    vdat = {}
    vdat["force"] = struct.unpack("I", fid.read(4))[0]
    vdat["line"] = struct.unpack("I", fid.read(4))[0]
    vdat["point"] = struct.unpack("I", fid.read(4))[0]
    vdat["sizeData"] = struct.unpack("I", fid.read(4))[0]
    vdat["ForceType"] = struct.unpack("I", fid.read(4))[0]
    vdat["pnt0"] = struct.unpack("I", fid.read(4))[0]
    vdat["pnt1"] = struct.unpack("I", fid.read(4))[0]
    vdat["pnt2"] = struct.unpack("I", fid.read(4))[0]
    vdat["dum"] = struct.unpack("2I", fid.read(8))[0]
    vdat["data"] = struct.unpack(str(vdat["sizeData"]) + "i", fid.read(4 * vdat["sizeData"]))

    return vdat


def local_readDEF(fid, address, _type):
    if address != -1:
        fid.seek(address, 0)

    [dumCRC, sizeDEF, typeDEF, dumMisc] = local_readARDFpointer(fid, -1)
    local_checkType(typeDEF, _type, fid)

    _def = {}
    _def["points"] = struct.unpack("I", fid.read(4))[0]
    _def["lines"] = struct.unpack("I", fid.read(4))[0]

    skip = 0
    if (typeDEF) == b"IDEF":
        skip = 96
    elif (typeDEF) == b"VDEF":
        skip = 144

    dum = struct.unpack(str(skip) + "s", fid.read(skip))[0]

    sizeText = 32
    _def["imageTitle"] = struct.unpack(str(sizeText) + "s", fid.read(sizeText))

    sizeHead = 16
    remainingSize = sizeDEF - 8 - skip - sizeHead - sizeText
    dum = struct.unpack(str(remainingSize) + "s", fid.read(remainingSize))[0]

    return _def


def local_readXDAT(fid, address):
    if address != -1:
        fid.seek(address, 0)

    [dumCRC, lastSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)

    if (lastType != b"XDAT") and (lastType != b"VSET"):
        print("ERROR: No XDAT or VSET here! Found " + str(lastType) + " Location:" + str(fid.tell() - 16))

    if lastType == b"XDAT":
        stepDist = lastSize - 16

        fid.seek(stepDist, 1)
    elif lastType == b"VSET":
        fid.seek(-16, 1)

    return []


def local_readTEXT(fid, loc):
    fid.seek(loc, 0)

    [dumCRC, dumSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)
    local_checkType(lastType, b"TEXT", fid)

    dumMisc = struct.unpack("I", fid.read(4))[0]
    sizeNote = struct.unpack("I", fid.read(4))[0]

    txt = struct.unpack(str(sizeNote) + "s", fid.read(sizeNote))[0]

    return txt


def local_readARDFpointer(fid, address):
    if address != -1:
        fid.seek(address, 0)

    checkCRC32 = struct.unpack("I", fid.read(4))[0]
    sizeBytes = struct.unpack("I", fid.read(4))[0]
    typePnt = struct.unpack("4s", fid.read(4))[0]
    miscNum = struct.unpack("I", fid.read(4))[0]

    return [checkCRC32, sizeBytes, typePnt, miscNum]


def local_checkType(found, test, fid):
    if found != test:
        print("ERROR: No " + str(test) + " here!  Found: " + str(found) + "  Location:" + str(fid.tell() - 16))


def readARDF(FN):
    D = {}
    D["FileName"] = FN.stem
    D["FileType"] = FN.suffix
    D["endNote"] = {}

    fid = open(FN, "rb")

    [dumCRC, dumSize, lastType, dumMisc] = local_readARDFpointer(fid, 0)
    local_checkType(lastType, b"ARDF", fid)

    F = {}

    F["ftoc"] = local_readTOC(fid, -1, b"FTOC")

    loc_TTOC = F["ftoc"]["sizeTable"] + 16
    F["ttoc"] = local_readTOC(fid, loc_TTOC, b"TTOC")

    F["ttoc"]["numbNotes"] = len(F["ttoc"]["pntText"])
    noteMain = local_readTEXT(fid, F["ttoc"]["pntText"][0])
    noteQuick = None
    noteThumb = None

    F["numbImag"] = len(F["ftoc"]["pntImag"])
    D["imageList"] = []
    D["y"] = []

    for n in range(F["numbImag"]):
        imagN = "imag" + str(n + 1)
        F[imagN] = local_readTOC(fid, F["ftoc"]["pntImag"][n], b"IMAG")

        loc_IMAG_TTOC = F["ftoc"]["pntImag"][n] + F[imagN]["sizeTable"]
        F[imagN]["ttoc"] = local_readTOC(fid, loc_IMAG_TTOC, b"TTOC")

        loc_IMAG_IDEF = F["ftoc"]["pntImag"][n] + F[imagN]["sizeTable"] + F[imagN]["ttoc"]["sizeTable"]
        F[imagN]["idef"] = local_readDEF(fid, loc_IMAG_IDEF, b"IDEF")

        D["imageList"].append(F[imagN]["idef"]["imageTitle"])

        idat = local_readTOC(fid, -1, b"IBOX")

        D["y"].append(idat["data"])

        [dumCRC, dumSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)
        local_checkType(lastType, b"GAMI", fid)

        numbImagText = len(F[imagN]["ttoc"]["pntText"])

        for r in range(numbImagText):
            theNote = local_readTEXT(fid, F[imagN]["ttoc"]["pntText"][r])

            if (numbImagText > 1) or (n == 0):
                if r == 0:
                    noteThumb = theNote
                elif r == 1:
                    F[imagN]["note"] = parseNotes(theNote)
                elif r == 2:
                    noteQuick = theNote
            else:
                F[imagN]["note"] = parseNotes(theNote)

    theNote = noteMain
    if noteQuick is not None:
        theNote = [noteMain, noteThumb, noteQuick]
    elif noteThumb is not None:
        theNote = [noteMain, noteThumb]
    D["notes"] = parseNotes(theNote)

    F["numbVolm"] = len(F["ftoc"]["pntVolm"])
    D["channelList"] = []

    idxZero = []
    incMin = []
    incMax = []

    for n in range(F["numbVolm"]):
        volmN = "volm" + str(n + 1)

        F[volmN] = local_readTOC(fid, F["ftoc"]["pntVolm"][n], b"VOLM")

        loc_VOLM_TTOC = F["ftoc"]["pntVolm"][n] + F[volmN]["sizeTable"]
        F[volmN]["ttoc"] = local_readTOC(fid, loc_VOLM_TTOC, b"TTOC")

        loc_VDEF_IMAG = F["ftoc"]["pntVolm"][n] + F[volmN]["sizeTable"] + F[volmN]["ttoc"]["sizeTable"]
        F[volmN]["vdef"] = local_readDEF(fid, loc_VDEF_IMAG, b"VDEF")

        F[volmN]["vchn"] = []
        F[volmN]["xdef"] = {}

        done = 0
        while done == 0:
            [dumCRC, lastSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)
            if lastType == b"VCHN":
                textSize = 32
                theChannel = struct.unpack(str(textSize) + "s", fid.read(textSize))[0]

                F[volmN]["vchn"].append(theChannel)

                remainingSize = lastSize - 16 - textSize
                dum = struct.unpack(str(remainingSize) + "s", fid.read(remainingSize))[0]
            elif lastType == b"XDEF":
                dum = struct.unpack("I", fid.read(4))[0]
                F[volmN]["xdef"]["sizeTable"] = struct.unpack("I", fid.read(4))[0]

                F[volmN]["xdef"]["text"] = struct.unpack(
                    str(F[volmN]["xdef"]["sizeTable"]) + "s",
                    fid.read(F[volmN]["xdef"]["sizeTable"]),
                )[0]

                dum = struct.unpack(
                    str(lastSize - 16 - 8 - F[volmN]["xdef"]["sizeTable"]) + "s",
                    fid.read(lastSize - 16 - 8 - F[volmN]["xdef"]["sizeTable"]),
                )[0]
                done = 1
            else:
                print("ERROR: " + str(lastType) + " not recognized!")
        D["channelList"] = [D["channelList"], F[volmN]["vchn"]]

        F[volmN]["idx"] = local_readTOC(fid, -1, b"VTOC")
        [dumCRC, lastSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)
        local_checkType(lastType, b"MLOV", fid)

        F[volmN]["line"] = {}

        for r in range(F[volmN]["vdef"]["lines"]):
            vsetN = "vset" + str(r + 1)
            loc = F[volmN]["idx"]["linPointer"][r]

            if loc != 0:
                F[volmN]["line"][vsetN] = local_readVSET(fid, loc)

                if F[volmN]["line"][vsetN]["line"] != r:
                    F[volmN]["scanDown"] = 1
                else:
                    F[volmN]["scanDown"] = 0

                if F[volmN]["line"][vsetN]["point"] == 0:
                    F[volmN]["trace"] = 1
                else:
                    F[volmN]["trace"] = 0
        try:
            idxZero = [i for i, val in enumerate(F[volmN]["idx"]["linPointer"]) if val == 0]
            incMin = 1
            incMax = 0

            if F[volmN]["scanDown"] == 1:
                idxZero = F[volmN]["vdef"]["lines"] - idxZero + 1
                incMin = 0
                incMax = 1
        except:
            pass
    if len(idxZero) > 0:
        idxZeroMin = min(idxZero) - incMin
        idxZeroMax = max(idxZero) + incMax
        for q in range(len(D["y"])):
            D["y"][q][idxZeroMin:idxZeroMax] = []
    D["endNote"]["IsImage"] = "1"
    D["FileStructure"] = F
    fid.close()

    return D


def getARDFdata(FN, getPoint=None, getLine=None, trace=None, fileStruct=None, verbose=False):
    hasFileStruct = 1
    if fileStruct is None:
        hasFileStruct = 0

    D = {}
    F = {}
    G = {}
    # FNbase = FN[:-4]
    FNmat = FN.with_suffix(".dat")  # FNbase + "dat"

    fid = open(FN, "rb")

    if hasFileStruct:
        F = fileStruct
        if verbose:
            print("FileStructure has been provided. Thank you!")
    elif os.path.exists(FNmat):
        tmp = open(FNmat, "rb")
        D = pickle.load(tmp)
        tmp.close()
        F = D["FileStructure"]
        if verbose:
            print("Found previous .DAT file. Loaded FileStructure!")
    else:
        if verbose:
            print("FileStructure will be regenerated from scratch.")

        [dumCRC, dumSize, lastType, dumMisc] = local_readARDFpointer(fid, 0)
        local_checkType(lastType, b"ARDF", fid)

        F["ftoc"] = local_readTOC(fid, -1, b"FTOC")
        F["numbVolm"] = len(F["ftoc"]["pntVolm"])

        D["channelList"] = []

        for n in range(F["numbVolm"]):
            volmN = "volm" + str(n + 1)

            F[volmN] = local_readTOC(fid, F["ftoc"]["pntVolm"][n], b"VOLM")

            loc_VOLM_TTOC = F["ftoc"]["pntVolm"][n] + F[volmN]["sizeTable"]
            F[volmN]["ttoc"] = local_readTOC(fid, loc_VOLM_TTOC, b"TTOC")

            loc_VDEF_IMAG = F["ftoc"]["pntVolm"][n] + F[volmN]["sizeTable"] + F[volmN]["ttoc"]["sizeTable"]
            F[volmN]["vdef"] = local_readDEF(fid, loc_VDEF_IMAG, b"VDEF")

            F[volmN]["vchn"] = []
            F[volmN]["xdef"] = {}

            done = 0

            while done == 0:
                [dumCRC, lastSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)
                if lastType == b"VCHN":
                    textSize = 32
                    theChannel = struct.unpack(str(textSize) + "s", fid.read(textSize))[0]

                    F[volmN]["vchn"].append(theChannel)

                    remainingSize = lastSize - 16 - textSize
                    dum = struct.unpack(str(remainingSize) + "s", fid.read(remainingSize))[0]
                elif lastType == b"XDEF":
                    dum = struct.unpack("I", fid.read(4))[0]
                    F[volmN]["xdef"]["sizeTable"] = struct.unpack("I", fid.read(4))[0]

                    F[volmN]["xdef"]["text"] = struct.unpack(
                        str(F[volmN]["xdef"]["sizeTable"]) + "s",
                        fid.read(F[volmN]["xdef"]["sizeTable"]),
                    )[0]

                    dum = struct.unpack(
                        str(lastSize - 16 - 8 - F[volmN]["xdef"]["sizeTable"]) + "s",
                        fid.read(lastSize - 16 - 8 - F[volmN]["xdef"]["sizeTable"]),
                    )[0]
                    done = 1
                else:
                    print("ERROR: " + str(lastType) + " not recognized!")
            D["channelList"] = [D["channelList"], F[volmN]["vchn"]]

            F[volmN]["idx"] = local_readTOC(fid, -1, b"VTOC")
            [dumCRC, lastSize, lastType, dumMisc] = local_readARDFpointer(fid, -1)
            local_checkType(lastType, b"MLOV", fid)

            F[volmN]["line"] = {}

            for r in range(F[volmN]["vdef"]["lines"]):
                vsetN = "vset" + str(r + 1)
                loc = F[volmN]["idx"]["linPointer"][r]

                if loc != 0:
                    F[volmN]["line"][vsetN] = local_readVSET(fid, loc)

                    if F[volmN]["line"][vsetN]["line"] != r:
                        F[volmN]["scanDown"] = 1
                    else:
                        F[volmN]["scanDown"] = 0

                    if F[volmN]["line"][vsetN]["point"] == 0:
                        F[volmN]["trace"] = 1
                    else:
                        F[volmN]["trace"] = 0

    getVolm = "volm1"
    if F["numbVolm"] > 1:
        if trace == F["volm1"]["trace"]:
            getVolm = "volm1"
        else:
            getVolm = "volm2"

    numbPoints = F[getVolm]["vdef"]["points"]
    numbLines = F[getVolm]["vdef"]["lines"]

    if F[getVolm]["scanDown"] == 1:
        if verbose:
            print(numbLines)
        if verbose:
            print(getLine)
        adjLine = numbLines - getLine - 1
    else:
        adjLine = getLine

    numbChannels = len(F["volm1"]["vchn"])

    locLine = F[getVolm]["idx"]["linPointer"][adjLine]

    if locLine != 0:
        fid.seek(locLine, 0)

        G["numbForce"] = []
        G["numbLine"] = []
        G["numbPoint"] = []
        G["locPrev"] = []
        G["locNext"] = []
        G["name"] = []
        G["y"] = []
        G["pnt0"] = []
        G["pnt1"] = []
        G["pnt2"] = []

        for n in range(numbPoints):
            vset = local_readVSET(fid, -1)
            G["numbForce"].append(vset["force"])
            G["numbLine"].append(vset["line"])
            G["numbPoint"].append(vset["point"])
            G["locPrev"].append(vset["prev"])
            G["locNext"].append(vset["next"])

            vnam = local_readVNAM(fid, -1)
            G["name"].append(vnam["name"])

            theData = []

            for r in range(numbChannels):
                vdat = local_readVDAT(fid, -1)
                theData.append(vdat["data"])

            local_readXDAT(fid, -1)

            G["y"].append(theData)
            G["pnt0"].append(vdat["pnt0"])
            G["pnt1"].append(vdat["pnt1"])
            G["pnt2"].append(vdat["pnt2"])
        if G["numbPoint"][0] != 0:
            G["numbForce"].reverse()
            G["numbLine"].reverse()
            G["numbPoint"].reverse()
            G["locPrev"].reverse()
            G["locNext"].reverse()
            G["y"].reverse()
            G["name"].reverse()
            G["pnt0"].reverse()
            G["pnt1"].reverse()
            G["pnt2"].reverse()

        if getPoint != -1:
            G["numbForce"] = G["numbForce"][getPoint]
            G["numbLine"] = G["numbLine"][getPoint]
            G["numbPoint"] = G["numbPoint"][getPoint]
            G["locPrev"] = G["locPrev"][getPoint]
            G["locNext"] = G["locNext"][getPoint]
            G["y"] = G["y"][getPoint]
            G["name"] = G["name"][getPoint]
            G["pnt0"] = G["pnt0"][getPoint]
            G["pnt1"] = G["pnt1"][getPoint]
            G["pnt2"] = G["pnt2"][getPoint]
    else:
        G = {}
    fid.close()
    return G


def extract_ardf(filename: Path) -> HyConvertedData:
    """Extract datas based on original Matlab source code written by Matthew Poss available here: https://ch.mathworks.com/matlabcentral/fileexchange/80212-ardf-to-matlab/?s_tid=LandingPageTabfx

    Parameters
    ----------
    filename : Path
        The name of the input file to be converted

    Returns
    -------
    HyConvertedData
    """

    gc.enable()

    F = readARDF(filename)
    data = {}
    attributes = {}
    metadata = {}
    for idx, channelName in enumerate(F["imageList"]):
        channelName = channelName[0].decode("utf8").split("\x00")[0]
        data[channelName] = F["y"][idx][:][:]

        attributes[channelName] = {}

        attributes[channelName]["name"] = channelName
        attributes[channelName]["shape"] = np.shape(F["y"][idx][:][:])
        attributes[channelName]["unit"] = "unknown"

        metadata[channelName] = {}
        for key, value in F["notes"][idx].items():
            metadata[channelName][key] = conversion_metadata(value)

    lines = F["FileStructure"]["volm1"]["vdef"]["lines"]
    points = F["FileStructure"]["volm1"]["vdef"]["points"]
    for idx, channelName in enumerate(F["FileStructure"]["volm1"]["vchn"]):
        channelName = channelName.decode("utf8").split("\x00")[0]

        max_size = 0
        channel_result = []
        for point in range(points):
            result_line = getARDFdata(filename, -1, point, 1, F["FileStructure"])
            result_line = list(np.array(result_line["y"], dtype=object)[:, idx])

            length = max(list(map(len, result_line)))

            if length > max_size:
                max_size = length

            result_line = list(
                map(
                    lambda elem: list(elem) + [np.nan] * (max_size - len(elem)),
                    result_line,
                )
            )

            channel_result.append(result_line)

        for line in range(lines):
            for point in range(points):
                if len(channel_result[line][point]) < max_size:
                    channel_result[line][point] = channel_result[line][point] + [np.nan] * (
                        max_size - len(channel_result[line][point])
                    )

        channel_result = np.array(channel_result)
        data[channelName] = channel_result
        attributes[channelName] = {}
        attributes[channelName]["name"] = channelName
        attributes[channelName]["shape"] = channel_result.shape

    gc.disable()
    extracted = HyConvertedData(data, metadata, attributes)
    return extracted
