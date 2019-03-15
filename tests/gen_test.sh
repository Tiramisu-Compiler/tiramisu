# Generates a new test out of an old test
# Example: ./gen_test.sh 10 143 "test 141 added"
export OLD=$1
export NEW=$2
if [ -f test_$NEW.cpp ]; then
    echo test_$NEW.cpp "already exists!"
    exit 1
fi
cp test_$OLD.cpp test_$NEW.cpp
cp wrapper_test_$OLD.cpp wrapper_test_$NEW.cpp
cp wrapper_test_$OLD.h wrapper_test_$NEW.h
vim test_$NEW.cpp
vim wrapper_test_$NEW.h
vim wrapper_test_$NEW.cpp
vim test_list.txt
vim README.md
cd ../build/
cmake ../
ctest --verbose -R $NEW
git add ../tests/test_$NEW.cpp ../tests/wrapper_test_$NEW.cpp ../tests/wrapper_test_$NEW.h test_list.txt README.md
if [ -n "$3" ]; then
    git commit -m "$3"
fi
cd -
