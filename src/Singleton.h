#ifndef SINGLETON_H
#define SINGLETON_H

#include <cstdlib>

/**
* Base template for singleton classes
*/
template <class T>
class Singleton
{
	private :
		static T* inst; // unique instance

		// Disalow any copy of a singleton class
		Singleton(const Singleton&);
		void operator =(const Singleton&);

	protected :
		Singleton() {}
		~Singleton() {}

	public :

		/**
		 * Return the unique instance of the class
		 */
		static T& instance()
		{
			if (!inst)
				inst = new T;

			return *inst;
		}

		/**
		 * Delete the unique instance of the class
		 */
		static void deleteInstance()
		{
			delete inst;
			inst = NULL;
		}
};

// Init of Singleton Instance
template <class T> T* Singleton<T>::inst = NULL;

#endif // SINGLETON_H
