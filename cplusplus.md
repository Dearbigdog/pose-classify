# cplusplus

#### sort()
>void sort (RandomAccessIterator first,     RandomAccessIterator last, Compare comp);

unstable 

<font color=green>
examples:
</font>

    std::sort (myvector.begin(), myvector.begin()+4);

#### to_string()
>string to_string (float val);
>
>string to_string (int val);

#### stoi(),stof(),stod()

>int stoi (const string&  str, size_t* idx = 0, int base = 10);

<font color=green>
examples:
</font>

    stoi ("0x7f",nullptr,0); //127
    
#### priority_queue
the first element is always the greatest.  
member: push,pop,top,size

#### make_pair
>pair<V1,V2> make_pair (T1&& x, T2&& y);

<font color=green>
examples:
</font>

    std::pair <int,int> foo;
    foo = std::make_pair (10,20);
    std::cout << foo.first << foo.second << '\n';

#### toupper/tolower
>int toupper ( int c );

Character to be converted, casted to an int, or EOF.Also we have isupper/islower