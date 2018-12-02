# File Structure
```
classification/
	- main.py
	- preprocess.py
	- data
	    train0.npy
	    train_label0.npy
	    val0.npy
	    val_label0.npy
	    test0.npy
	    test_label0.npy
	    - images1
	        - ...
	    - images2
	        - ...
	    ...
	    - images12
	        - ...
report/
	- `main.py`
	- `preprocess.py`
	- `data/`
		`data.npy` 			# all images
		`indications.npy`	# (reasons for viewing doc)
		`findings.npy`		# (preliminary findings)
		`impressions.npy`	# (conclusion)
		
		`img_names`			# (image names in the same sequential order as in `data.npy`
		`mapping.js` 		# (map from image names to xml report file names)
		`targets.js`		# (map from xml report file names to triplet of indication, finding, and impression)

```
