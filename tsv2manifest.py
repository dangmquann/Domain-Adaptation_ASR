import os
import sys
import csv
import glob
import tqdm
import time
import random
import multiprocessing

import lhotse
from lhotse import MultiCut, CutSet
from lhotse import SupervisionSegment
from lhotse.supervision import AlignmentItem


tsv_path = sys.argv[1]
njobs = int(sys.argv[2])
manifest_path = sys.argv[3]
root_path=sys.argv[4] if len(sys.argv) >= 5 else None

rows = []
with open(tsv_path, encoding='utf-8') as fin:
    reader = csv.reader(fin, delimiter='\t')
    header = next(reader, None)
    for row in reader:
        if len(str(row[-1]).strip()) < 1:
            print(f'ignore {row}')
            continue
        if root_path is not None:
            row[0] = os.path.join(root_path, row[0])
        if len(row) == 2:
            row = [row[0], 0, row[1]]
        rows.append(row)

print('length of total files:', len(rows))

cuts = []

def load_cut(args):
    idx, row = args
    audio_file = row[0]
    text_input = str(row[2]).replace("?", "").replace(".","").replace(",","").replace("!","").replace("-","")
    # prompt = random.choice(PROMPTS)
    if (idx % 5000) == 0:
        sys.stderr.write(str(idx) + ' ... ')
        sys.stderr.flush()
    try:
        cut = lhotse.Recording.from_file(audio_file).to_cut()
        if isinstance(cut, MultiCut):
            cut = cut.to_mono()[0]
        cut.id = f"{cut.id}-{idx}"
        cut.supervisions = [SupervisionSegment(
            id=cut.recording_id, recording_id=cut.recording_id,
            start=cut.start, duration=cut.duration, channel=cut.channel,
            text=text_input
        )]
        
        # alignment={
        #     'words': [
        #         AlignmentItem(symbol='AY0', start=33.0, duration=0.6),
        #         AlignmentItem(symbol='S', start=33.6, duration=0.4)
        #     ]
        # }
        
        return cut
    except Exception as e:
        print(e)
        return None

params = []
for idx, row in enumerate(rows):
    params.append((idx, row))

pool = multiprocessing.Pool(njobs)
try:
    start = time.time()
    cuts = pool.map_async(load_cut, params).get(9999999)
    sys.stderr.write('\n' + str(time.time() - start) + '\n')

except KeyboardInterrupt:
    sys.stderr.write('\n')
    pool.terminate()
    sys.exit(1)

sys.stderr.write('\n')
pool.close()
pool.join()

cuts = [cut for cut in cuts if cut is not None]
cuts = CutSet.from_cuts(cuts)
cuts.to_file(manifest_path)
