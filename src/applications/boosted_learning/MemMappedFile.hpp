/// 2010 jhbecker, created: 2010.03.05
/// 2011 mmathias

#ifndef MEMMAPPEDFILE_HH
#define MEMMAPPEDFILE_HH


#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include <sys/mman.h>
#include <fcntl.h>
#include <sys/sysinfo.h>

#include <iostream>
#include <cassert>
#include <stdexcept>
#include <string>

namespace jhb
{

namespace fs = boost::filesystem;

template<typename T>
class MemMappedFile
{
protected:
    std::string m_filename;
    long unsigned m_num_elements;
    long unsigned int m_filesize;
    T *m_data;
    int m_fd;
    bool m_valid;
    bool m_inMem;

public:
    typedef boost::shared_ptr<MemMappedFile<T> > shared_ptr;

    enum access_mode { READ, WRITE };
    enum data_saved {FILE, MEM};


    /// Implementation based on
    /// http://stackoverflow.com/questions/349889
    /// http://linux.die.net/man/2/sysinfo
    std::size_t get_memory_available()
    {
        struct sysinfo info;

#if defined(NDEBUG)
        sysinfo(&info);
#else
        const int ret = sysinfo(&info);
        assert(ret == 0);
#endif
        return info.totalram * info.mem_unit;
    }

    MemMappedFile()
        : m_filename(""), m_num_elements(0),
          m_filesize(0),
          m_data(NULL), m_fd(-1), m_valid(false), m_inMem(false)
    {}

    int open_mmap(const std::string &filename, long unsigned num_elements, access_mode mode = READ)
    {

        m_filename = filename;
        m_num_elements = num_elements;
        m_filesize =  m_num_elements * sizeof(T);
        std::cout << "Requested memory: " << m_filesize / (1024.0f*1024)
                  << " MBytes, available memory: " << (get_memory_available() * 0.8) / (1024.0f*1024)
                  << " MBytes" << std::flush << std::endl;
        const size_t mem = int(get_memory_available() * 0.6);

        if (mem > m_filesize)
        {
            std::cout << "Data fits into memory!" << std::flush << std::endl << std::endl;
            m_data = new T [num_elements];
            m_inMem = true;
            m_valid = true;
            return MEM;
        }
        else
        {
            std::cout << "Data does _not_ fit into memory. Using mmap file." << std::flush << std::endl << std::endl;
        }

        if (mode == READ)
        {
            if (!fs::exists(fs::path(filename)))
            {
                throw std::runtime_error("Input file '" + filename + "' does not exist.");
            }

            // prepare the mmap input file
            m_fd = open(filename.c_str(), O_RDONLY);

            if (m_fd == -1)
            {
                throw std::runtime_error("Error opening the file '" + filename + "' for reading.");
            }

            // map the file
            m_data = (T *)mmap(0, m_filesize, PROT_READ, MAP_SHARED, m_fd, 0);

            if (m_data == MAP_FAILED)
            {
                close(m_fd);
                throw std::runtime_error("Error mapping the file '" + m_filename + "'.");
            }
        }
        else
        {
            // prepare the mmap output file
            int m_fd = open(m_filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);

            if (m_fd == -1)
            {
                throw std::runtime_error("Error opening file '" + m_filename + "' for writing!");
            }

            // seek to desired filesize
            off_t result = lseek(m_fd, m_filesize - 1, SEEK_SET);

            if (result == -1)
            {
                throw std::runtime_error("Error calling lseek() to 'stretch' the file '" + m_filename + "' to the desired size.");
            }

            // write one empty string s.t. the file gets indeed the desired filesize
            result = write(m_fd, "", 1);

            if (result != 1)
            {
                close(m_fd);
                throw std::runtime_error("Error writing last byte of the file '" + m_filename + "'.");
            }

            // map the file
            m_data = (T *)mmap(0, m_filesize, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);

            if (m_data == MAP_FAILED)
            {
                close(m_fd);
                throw std::runtime_error("Error mapping the file '" + m_filename + "'.");
            }
        }

        //close(m_fd);
        m_valid = true;
        return FILE;
    }

    void close_mmap()
    {
        if (m_valid)
        {
            if (m_inMem)
            {
                m_inMem = false;
                delete[] m_data;
            }
            else
            {
                // unmap the file
                if (munmap(m_data, m_filesize) == -1)
                {
                    throw std::runtime_error("Error un-mapping the file '" + m_filename + "'");
                }

                // unmapping does not close the file pointer
                if (m_fd != -1)
                {
                    close(m_fd);
                }

                m_fd = -1;
            }

            m_valid = false;
        }
    }

    ~MemMappedFile()
    {
        close_mmap();
    }

    inline T &operator[](unsigned long Index)
    {
        return(m_data[Index]);
    }

    inline T operator[](unsigned long Index) const
    {
        return(m_data[Index]);
    }

    inline unsigned long size() const
    {
        return m_num_elements;
    }
};

} // namespace jhb

#endif // MEMMAPPEDFILE_HH

