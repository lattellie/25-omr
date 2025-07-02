# String Quartet OMR


- Input: PDF + JSON file specifying where the time signature shifts were
- Output: MusicXML file

## Prequisites

- Python 3.9
- Optional: NVidia GPU with CUDA 12.4

## Set Up

- Clone the repository
- Install dependencies using `pip install -r requirements.txt`
    If having issues, try using `pip install -r requirements-no-versions.txt`

## Test run
- Demo score:
  - run the pdf2musicXML.py, which will run on beethoven1's string quartet (found in string_dataset/pdf_data/beethoven1.pdf)
  - output can be found in string_dataset/output once the model finishes execution.
- Test your own score:
    - Add your pdf in pdf_data/[piece_name][piece_name].pdf (for instance, if your pdf is called mozart1, save it in `pdf_data/mozart1/mozart1.pdf`)
    - Add the specification of where the time signature shifts were to a new .json file, template can be found in jsonTemplate.json, save it in pdf_data/[piece_name][piece_name].json (for instance, if your pdf is called mozart1, save it in `pdf_data/mozart1/mozart1.json`)
    - Add the piece name in string_dataset/piecesToRun, for instance ['mozart1']
    - JSON file explained
    ```python
        {
    "numPage": 20,  # how many pages are in the pdf
    "numPerPage": 1, # default 1 
    "rotate": false, # default false
    "tsChange": [
        # MUST have the fist one
        {
        "page": 1,
        "loc":[0,0],
        "time_signature": [4,4], # the inital time signature
        "mov": 1 # not used for now
        },
        # on page 7, the first line fisrt bar, 
        # time signature changed to 2/4
        {
        "page": 7,
        "loc":[0,0],
        "time_signature": [2,4],
        "mov":2
        },
        # on page 9, the 4th line first bar, 
        # time signature changed to 3/4
        {
        "page": 9,
        "loc":[3,0],
        "time_signature": [3,4],
        "mov":3
        },
        # ... etc.
        {
        "page": 11,
        "loc":[2,0],
        "time_signature": [12,8],
        "mov":3
        }
    ],
    "numTrack": 4, # how many instruments are there
    "track_shift": [0,0,0,0], # not used for now (for transposing instruments, not used in string quartet)
    "clef_options": [[1],[1],[0],[-1,-2]]
    # what are the possible clefs in that track
    # 1: treble, 0: alto (viola), -1: bass, -2: tenor (cello)
    # [[first track options],[2nd track options],[3rd ...], [4th...]]
    }
    ```


This project utilized the segmentation models from [oemer](https://github.com/BreezeWhite/oemer)