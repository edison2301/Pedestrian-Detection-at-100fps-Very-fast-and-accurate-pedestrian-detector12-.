#!/usr/bin/sh
# Do not use this script unless you know what you are doing,
# start by reading readme.text to know which are the first step to do get this code base to compile.

# change num threads depending on available RAM and Cpu cores.
NUM_THREADS=15

while true; do
		read -p "Do you _really_ know what you are doing ? [y/N]" yn
		case $yn in
		[Yy]* ) fn_function;;
		[Nn]* ) exit ;;
		* ) echo "Please answer yes or no.";;
		esac
	done

for dirname in ls -d src/applications/*/ src/tests/*/
  cd ${dirname}
  cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo .
  make -j${NUM_THREADS}
done

