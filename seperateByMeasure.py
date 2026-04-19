from music21 import converter, note, meter, key, clef, stream
from music21 import spanner

ESSENTIAL_CLASSES = (
    note.Note,
    note.Rest,
    meter.TimeSignature,
    key.KeySignature,
    clef.Clef,
)

def keep_only_essential(s):
    new_stream = stream.Score()
    for part in s.parts:
        new_part = stream.Part()
        for m in part.getElementsByClass('Measure'):
            new_m = stream.Measure(number=m.number)
            for el in m:
                if isinstance(el, ESSENTIAL_CLASSES):
                    new_m.append(el)
            new_part.append(new_m)
        new_stream.append(new_part)
    return new_stream

def remove_spanners(stream_obj):
    for sp in list(stream_obj.recurse().getElementsByClass(spanner.Spanner)):
        stream_obj.remove(sp)

def seperateMeasure(filePath, beginningIndex):
    score = converter.parse(filePath)
    for i in range(len(beginningIndex) - 1):
        start = beginningIndex[i]
        end = beginningIndex[i + 1] - 1  # stop before next boundary
        segmentOri = score.measures(start, end)
        segment = keep_only_essential(segmentOri)
        remove_spanners(segment)

        # write to new file
        segment.write("musicxml", f"{filePath.replace('.musicxml',f'_{i+1}.musicxml')}")
        print(f"finish parsing score {i+1}")
    last_start = beginningIndex[-1]
    last_segment = score.measures(last_start, None)
    last_segment.write("musicxml", f"{filePath.replace('.musicxml',f'_{i+2}.musicxml')}")
    print(f"finish parsing score {i+1}")

if __name__ == '__main__':
    dataSeperate = [1, 12, 33, 53, 77, 98, 119, 141, 160, 170, 181, 194,
        223, 248, 267, 287, 309, 333, 356, 376, 386, 397,
        408, 418, 429, 441, 455, 469, 480, 491]
    seperateMeasure(rf"C:\Ellie\APIs\25-omr\orch_dataset\Bee_5_challenge\val2\Bee_5_mvt1.musicxml",beginningIndex=dataSeperate)