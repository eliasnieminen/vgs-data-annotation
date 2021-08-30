create_subdirs() {
	mkdir crosstask
	mkdir crosstask/train
	mkdir crosstask/val

	mkdir youcook2
	mkdir youcook2/train
	mkdir youcook2/test
	mkdir youcook2/val
	
	mkdir howto100m
	mkdir howto100m/train
}

create_frame_dirs() {
	mkdir crosstask/train/frames
	mkdir crosstask/val/frames
	
	mkdir youcook2/train/frames
	mkdir youcook2/test/frames
	mkdir youcook2/val/frames
	
	mkdir howto100m/train/frames	
}

mkdir data/

mkdir download/
cd download
create_subdirs
cd ..

mkdir clips
cd clips
create_subdirs
create_frame_dirs
cd ..

mkdir clip_annotations
cd clip_annotations
create_subdirs
cd ..

