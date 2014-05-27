#ifndef DATASEQUENCE_HPP
#define DATASEQUENCE_HPP

#include "DataSequenceHeader.pb.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <boost/filesystem.hpp>
#include <boost/cstdint.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/foreach.hpp>

#include <fstream>
#include <string>
#include <map>

#include <stdexcept>
#include <utility> // std::make_pair


namespace doppia
{

/// This class allows to read and write sequence data files
/// These files store a sequence of protocol buffer messages
/// All messages but the first should be of the same type
/// The first message is always a DataSequenceHeader message
///
/// This class is one solution for the problem of
/// storing a sequence of protocol buffer messages
///
/// The implementation is based on
///
/// @see http://code.google.com/p/protobuf/
/// @see http://code.google.com/apis/protocolbuffers/docs/techniques.html#large-data
/// @see DataSequenceHeader.proto
template<typename DataType>
class DataSequence
{
public:

    typedef DataType data_type;
    typedef std::map<std::string, std::string> attributes_t;

    /// defaults to read mode
    DataSequence(std::string filename);
    DataSequence(std::string filename, std::ios::openmode mode);

    /// defaults to write mode
    DataSequence(std::string filename, const attributes_t &attributes);
    DataSequence(std::string filename, std::ios::openmode mode, const attributes_t &attributes);

    ~DataSequence();

    const attributes_t &get_attributes();

    /// Read the new message
    void read(DataType &data);

    /// Write one more message into the file
    void write(const DataType &data);

    /// alias for read
    void operator>>(DataType &data);

    /// alias for write
    void operator<<(const DataType &data);

    /// flush all pending data to disk
    void flush();

protected:

    void init(std::string filename, std::ios::openmode mode, const attributes_t &attributes);

    void read_header();
    void write_header();

    attributes_t attributes;
    boost::scoped_ptr<std::fstream> file_stream_p;

    boost::scoped_ptr<google::protobuf::io::ZeroCopyInputStream> input_stream_p;
    boost::scoped_ptr<google::protobuf::io::CodedInputStream> input_coded_stream_p;

    boost::scoped_ptr<google::protobuf::io::ZeroCopyOutputStream> output_stream_p;
    boost::scoped_ptr<google::protobuf::io::CodedOutputStream> output_coded_stream_p;

};


template<typename DataType>
DataSequence<DataType>::DataSequence(std::string filename)
{
    const attributes_t empty_attributes;
    init(filename, std::ios::in, empty_attributes);
    return;
}


template<typename DataType>
DataSequence<DataType>::DataSequence(std::string filename, std::ios::openmode mode)
{
    const attributes_t empty_attributes;
    init(filename, mode, empty_attributes);
    return;
}


template<typename DataType>
DataSequence<DataType>::DataSequence(std::string filename, const attributes_t &attributes)
{
    init(filename, std::ios::out, attributes);
    return;
}

template<typename DataType>
DataSequence<DataType>::DataSequence(std::string filename, std::ios::openmode mode, const attributes_t &attributes)
{
    init(filename, mode, attributes);
    return;
}


template<typename DataType>
void DataSequence<DataType>::init(std::string filename, std::ios::openmode mode, const attributes_t &the_attributes)
{
    using std::ios;


    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;


    // open the file --
    if(mode == std::ios::in)
    {
        if(the_attributes.empty() == false)
        {
            throw std::invalid_argument("In read mode DataSequence<DataType> ignores the attributes given to the constructor");
        }

        if(boost::filesystem::exists(filename) == false)
        {
            throw std::runtime_error("Could not find the indicated DataSequence file");
        }

        file_stream_p.reset(new std::fstream(filename.c_str(), ios::in | ios::binary));

        using google::protobuf::io::IstreamInputStream;
        using google::protobuf::io::CodedInputStream;

        input_stream_p.reset(new IstreamInputStream(file_stream_p.get()));
        input_coded_stream_p.reset(new CodedInputStream(input_stream_p.get()));

        read_header();
    }
    else if(mode == std::ios::out)
    {
        file_stream_p.reset(new std::fstream(filename.c_str(), ios::out | ios::binary | ios::trunc));

        using google::protobuf::io::OstreamOutputStream;
        using google::protobuf::io::CodedOutputStream;

        output_stream_p.reset(new OstreamOutputStream(file_stream_p.get()));
        output_coded_stream_p.reset(new CodedOutputStream(output_stream_p.get()));

        attributes = the_attributes;
        write_header();
    }
    else
    {
        throw std::invalid_argument("DataSequence constructor received an unmanaged openmode");
    }

    return;
} // end of DataSequence<DataType>::init

template<typename DataType>
void DataSequence<DataType>::read_header()
{
    using doppia_protobuf::DataSequenceHeader;

    assert(input_stream_p);

    DataSequenceHeader header;

    boost::uint64_t size;
    bool success = input_coded_stream_p->ReadLittleEndian64(&size);

    const bool use_zero_copy_stream = false;
    if(use_zero_copy_stream)
    {
        success &=
                header.ParseFromBoundedZeroCopyStream(input_stream_p.get(), static_cast<int>(size));
    }
    else
    { // work around
        std::string buffer;
        input_coded_stream_p->ReadString(&buffer, size);
        success &= header.ParseFromString(buffer);
    }

    if (success == false)
    {
        //printf("ERROR: '%s'\n", header.InitializationErrorString().c_str());
        throw std::runtime_error("Failed to parse the DataSequenceHeader");
    }

    // set the attributes map --
    for(int i =0; i < header.attributes_size(); i+=1)
    {
        using doppia_protobuf::DataSequenceAttribute;
        const DataSequenceAttribute &attribute = header.attributes(i);

        attributes.insert(std::make_pair(attribute.name(), attribute.value()));
    }

    return;
}

template<typename DataType>
void DataSequence<DataType>::write_header()
{
    using doppia_protobuf::DataSequenceHeader;
    using doppia_protobuf::DataSequenceAttribute;

    assert(output_stream_p);

    DataSequenceHeader header;
    BOOST_FOREACH(const attributes_t::value_type &attribute, attributes)
    {
        DataSequenceAttribute *attribute_p = header.add_attributes();

        attribute_p->set_name(attribute.first);
        attribute_p->set_value(attribute.second);
    }
    boost::uint64_t size = header.ByteSize();
    output_coded_stream_p->WriteLittleEndian64(size);
    //const bool success = header.SerializeToZeroCopyStream(output_stream_p.get());
    const bool success = header.SerializeToCodedStream(output_coded_stream_p.get());

    if (success == false or output_coded_stream_p->HadError())
    {
        throw std::runtime_error("Failed to write the DataSequenceHeader");
    }

    return;
}


template<typename DataType>
DataSequence<DataType>::~DataSequence()
{
    // file manipulation objects are interlinked,
    // so the destruction needs to be in a specific order
    input_coded_stream_p.reset();
    input_stream_p.reset();

    output_coded_stream_p.reset();
    output_stream_p.reset();

    file_stream_p->close();
    file_stream_p.reset();
    return;
}

template<typename DataType>
void DataSequence<DataType>::flush()
{
    file_stream_p->flush();
    return;
}

template<typename DataType>
const typename DataSequence<DataType>::attributes_t & DataSequence<DataType>::get_attributes()
{
    return attributes;
}

template<typename DataType>
void DataSequence<DataType>::operator>>(DataType &data)
{
    read(data);
    return;
}


template<typename DataType>
void DataSequence<DataType>::read(DataType &data)
{
    boost::uint64_t size;
    const bool read_size_success = input_coded_stream_p->ReadLittleEndian64(&size);

    bool read_data_success = false;

    const bool use_zero_copy_stream = false;
    if(use_zero_copy_stream)
    {
        read_data_success =
                    data.ParseFromBoundedZeroCopyStream(input_stream_p.get(), static_cast<int>(size));
    }
    else
    { // work around
        std::string buffer;
        input_coded_stream_p->ReadString(&buffer, size);
        read_data_success = data.ParseFromString(buffer);
    }

    if (read_size_success == false or read_data_success == false)
    {
        throw std::runtime_error("Failed to read a data message during DataSequence<DataType>::read");
    }
    return;
}

template<typename DataType>
void DataSequence<DataType>::operator<<(const DataType &data)
{
    write(data);
    return;
}


template<typename DataType>
void DataSequence<DataType>::write(const DataType &data)
{
    boost::uint64_t size = data.ByteSize();
    output_coded_stream_p->WriteLittleEndian64(size);
    //const bool success = data.SerializeToZeroCopyStream(output_stream_p.get());
    //const bool success = data.SerializeToOstream(file_stream_p.get());
    const bool success = data.SerializeToCodedStream(output_coded_stream_p.get());

    if (success == false or output_coded_stream_p->HadError())
    {
        throw std::runtime_error("Failed to write a data message during DataSequence<DataType>::write");
    }

    return;
}


} // end of namespace doppia

#endif // DATASEQUENCE_HPP
